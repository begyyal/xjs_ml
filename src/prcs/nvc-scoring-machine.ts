import { Array2, TimeUnit, toMsec, UArray } from "xjs-common";
import { MlHelper } from "../func/ml-helper";

type Point = { id: number, timestamp?: number };
type ValueWeight = Record<string, Point[]>;
export type NvcModel<Props extends Record<string, string | number>> = Record<keyof Props, ValueWeight>;
export type NvcDataset<Cls extends string, Props extends Record<string, string | number>> = Record<Cls, NvcModel<Props>>;
export class NvcScoringMachine<Cls extends string, Props extends Record<string, string | number>> {
    private readonly _remainingMsec: number;
    private readonly _trimmingScale: number;
    private _idCounter = 0;
    private _dataset = {} as NvcDataset<Cls, Props>;
    get dataset() { return this._dataset; }
    set dataset(d: NvcDataset<Cls, Props>) {
        this._idCounter = NvcScoringMachine.flatDataset2points(d).map(p => p.id).sort((a, b) => b - a)?.[0] ?? 0;
        this._dataset = d;
    }
    constructor(op?: { remainingMsec?: number, trimmingScale?: number }) {
        this._remainingMsec = MlHelper.checkNumericArgument(op?.remainingMsec, 0, null, { includeLower: false, d: toMsec(1, TimeUnit.Day) });
        this._trimmingScale = MlHelper.checkNumericArgument(op?.trimmingScale, 1, null, { d: 10000 });
    }
    train(r: { props: Props, cls: Cls, timestamp?: number }): void {
        const model = ((this._dataset[r.cls] as any) ??= {}) as NvcModel<Props>;
        Object.entries(r.props).forEach(e => {
            (model as any)[e[0]] ??= {}; // typescript is broken.
            (model[e[0]][e[1].toString()] ??= []).push({ id: ++this._idCounter, timestamp: r.timestamp ?? -1 })
        });
    }
    /**
     * consider any class that doesn't exist as "0".
     */
    score(input: Props, op?: { thresholdTime?: number, target?: "probability" | "likelihood", cls?: Cls[] }): Partial<Record<Cls, number>> {
        const _thresholdTime = op?.thresholdTime ?? Date.now() - this._remainingMsec;
        const _target = op?.target ?? "probability";
        const _classes = op?.cls ?? Object.keys(this._dataset) as Cls[];
        if (_classes.length === 0) return {};
        const weight2point = (w: { id: number, timestamp?: number }[] = []) =>
            w.map(({ timestamp }) => !timestamp || timestamp < 0 ? 1 :
                timestamp - _thresholdTime <= 0 ? 0 : (timestamp - _thresholdTime) / this._remainingMsec).reduce((a, b) => a + b, 0);
        const entries = Object.entries(input);
        const sumCount = (m: NvcModel<Props>) => m?.[entries[0][0]] ? weight2point(Object.values(m[entries[0][0]]).flat()) : 0;
        const countSet = Array2.record(_classes, { vgen: k => sumCount(this._dataset[k]) });
        const calcL = (m: NvcModel<Props>, count: number) =>
            !m || count === 0 ? 0 : entries.map(e => {
                const num = weight2point(m[e[0]][e[1].toString()])
                return (num + 1) / (count + entries.length);
            }).reduce((a, b) => a * b);
        if (_target === "likelihood") return Array2.record(_classes, { vgen: k => calcL(this._dataset[k], countSet[k]) })
        const totalCount = Array2.sum(Object.values(countSet));
        const calcP = (m: NvcModel<Props>, count: number) => {
            const l = calcL(m, count);
            return l === 0 ? 0 : l * count / totalCount;
        }
        const pSet = Array2.record(_classes, { vgen: k => calcP(this._dataset[k], countSet[k]) });
        const denom = Array2.sum(Object.values(pSet));
        return Array2.record(_classes, { vgen: k => pSet[k] === 0 ? 0 : pSet[k] / denom });
    }
    rank(valueSet: Record<keyof Props, (string | number)[]>, expected: Cls): [number, Props][] {
        const entries = Object.entries(valueSet), thresholdTime = Date.now() - this._remainingMsec;
        const expand2combinations: (i: number, o?: Props) => Props[] = (i, o = {} as Props) =>
            i === entries.length ? [o] : entries[i][1].flatMap(v => expand2combinations(i + 1, { ...o, [entries[i][0]]: v }));
        return expand2combinations(0)
            .map(p => [this.score(p, { thresholdTime, target: "likelihood", cls: [expected] })[expected] ?? 0, p] as [number, Props])
            .sort((a, b) => b[0] - a[0]);
    }
    flush(op?: { thresholdTime?: number }): void {
        const _thresholdTime = op?.thresholdTime ?? Date.now() - this._remainingMsec;
        Object.values(this._dataset)
            .flatMap(m => Object.values(m).flatMap((c2e: ValueWeight) => Object.values(c2e)))
            .forEach(exps => UArray.takeOut(exps, ({ id, timestamp }) => timestamp > 0 && timestamp < _thresholdTime || id <= this._idCounter - this._trimmingScale));
    }
    static adjustTimestamp<Cls extends string, P extends Record<string, string | number>>(dataset: NvcDataset<Cls, P>, converter: (original: number) => number): void {
        NvcScoringMachine.flatDataset2points(dataset).filter(v => v.timestamp).forEach(v => v.timestamp = converter(v.timestamp));
    }
    static shiftTimestampToNow<Cls extends string, P extends Record<string, string | number>>(dataset: NvcDataset<Cls, P>): void {
        const latestTs = NvcScoringMachine.flatDataset2points(dataset).map(v => v.timestamp).filter(ts => ts).sort((a, b) => b - a)[0], now = Date.now();
        if (latestTs) NvcScoringMachine.adjustTimestamp(dataset, ts => ts + now - latestTs);
    }
    static flatDataset2points<Cls extends string, P extends Record<string, string | number>>(dataset: NvcDataset<Cls, P>): Point[] {
        return Object.values(dataset).flatMap((e: NvcModel<P>) => Object.values(e).flatMap(e2 => Object.values(e2)).flat());
    }
}