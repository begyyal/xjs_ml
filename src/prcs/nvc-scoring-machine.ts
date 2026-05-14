import { Array2, TimeUnit, toMsec, UArray } from "xjs-common";
import { MlHelper } from "../func/ml-helper";

type ValueWeight = Record<string, { id: number, timestamp?: number }[]>;
export type NvcModel<Props extends Record<string, string | number>> = Record<keyof Props, ValueWeight>;
export type NvcDataset<Cls extends string, Props extends Record<string, string | number>> = Record<Cls, NvcModel<Props>>;
export class NvcScoringMachine<Cls extends string, Props extends Record<string, string | number>> {
    private readonly _remainingMsec: number;
    private readonly _trimmingScale: number;
    private _idCounter = 0;
    private _dataset = {} as NvcDataset<Cls, Props>;
    constructor(op?: { remainingMsec?: number, trimmingScale?: number }) {
        this._remainingMsec = MlHelper.checkNumericArgument(op?.remainingMsec, 0, null, { includeLower: false, d: toMsec(1, TimeUnit.Day) });
        this._trimmingScale = MlHelper.checkNumericArgument(op?.trimmingScale, 1, null, { d: 10000 });
    }
    train(r: { props: Props, cls: Cls, timestamp?: number }): void {
        const model = this._dataset[r.cls];
        Object.entries(r.props).forEach(e => {
            (model as any)[e[0]] ??= {}; // typescript is broken.
            (model[e[0]][e[1].toString()] ??= []).push({ id: ++this._idCounter, timestamp: r.timestamp ?? -1 })
        });
    }
    /**
     * consider any class that doesn't exist as "0".
     */
    score(input: Props, thresholdTime: number = Date.now() - this._remainingMsec): Partial<Record<Cls, number>> {
        const classes = Object.keys(this._dataset) as Cls[];
        if (classes.length === 0) return {};
        const weight2point = (w: { id: number, timestamp?: number }[] = []) =>
            w.map(({ timestamp: exp }) => !exp || exp < 0 ? 1 :
                exp - thresholdTime <= 0 ? 0 : (exp - thresholdTime) / this._remainingMsec).reduce((a, b) => a + b, 0);
        const entries = Object.entries(input);
        const sumCount = (m: NvcModel<Props>) => m[entries[0][0]] ? weight2point(Object.values(m[entries[0][0]]).flat()) : 0;
        const countSet = Array2.record(classes, { vgen: k => sumCount(this._dataset[k]) });
        const totalCount = Array2.sum(Object.values(countSet));
        const calcP = (m: NvcModel<Props>, count: number) =>
            count === 0 ? 0 : entries.map(e => {
                const num = weight2point(m[e[0]][e[1].toString()])
                return (num + 1) / (count + classes.length);
            }).reduce((a, b) => a * b) * count / totalCount;
        const pSet = Array2.record(classes, { vgen: k => calcP(this._dataset[k], countSet[k]) });
        const denom = Array2.sum(Object.values(pSet));
        return Array2.record(classes, { vgen: k => pSet[k] === 0 ? 0 : pSet[k] / denom });
    }
    rank(valueSet: Record<keyof Props, (string | number)[]>): [number, Props][] {
        const entries = Object.entries(valueSet), thresholdTime = Date.now() - this._remainingMsec;
        const expand2combinations: (i: number, o?: Props) => Props[] = (i, o = {} as Props) =>
            i === entries.length ? [o] : entries[i][1].flatMap(v => expand2combinations(i + 1, { ...o, [entries[i][0]]: v }));
        return expand2combinations(0).map(p => [this.score(p, thresholdTime), p] as [number, Props]).sort((a, b) => a[0] - b[0]);
    }
    flush(): void {
        const now = Date.now();
        Object.values(this._dataset)
            .flatMap(m => Object.values(m).flatMap((c2e: ValueWeight) => Object.values(c2e)))
            .forEach(exps => UArray.takeOut(exps, ({ id, timestamp: exp }) => exp > 0 && exp < now || id <= this._idCounter - this._trimmingScale));
    }
    static adjustTimestamp<Cls extends string, P extends Record<string, string | number>>(dataset: NvcDataset<Cls, P>, converter: (original: number) => number): void {
        Object.values(dataset)
            .flatMap((e: NvcModel<P>) => Object.values(e).flatMap(e2 => Object.values(e2)).flat())
            .filter(v => v.timestamp).forEach(v => v.timestamp = converter(v.timestamp));
    }
    static shiftTimestampToNow<Cls extends string, P extends Record<string, string | number>>(dataset: NvcDataset<Cls, P>): void {
        const latestTs = Object.values(dataset)
            .flatMap((e: NvcModel<P>) => Object.values(e).flatMap(e2 => Object.values(e2)).flat())
            .map(v => v.timestamp).filter(ts => ts).sort((a, b) => b - a)[0], now = Date.now();
        if (latestTs) NvcScoringMachine.adjustTimestamp(dataset, ts => ts + now - latestTs);
    }
}