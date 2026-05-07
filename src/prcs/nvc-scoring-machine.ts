import { TimeUnit, toMsec, UArray } from "xjs-common";
import { MlHelper } from "../func/ml-helper";

type ValueWeight = Record<string, { id: number, timestamp?: number }[]>;
type Model<Props extends Record<string, string | number>> = Record<keyof Props, ValueWeight>;
export class NvcScoringMachine<Props extends Record<string, string | number>> {
    private readonly _defaultScore: number;
    private readonly _remainingMsec: number;
    private readonly _trimmingScale: number;
    private _idCounter = 0;
    private _positive: Model<Props> = {} as Model<Props>;
    private _negative: Model<Props> = {} as Model<Props>;
    get models() { return { positive: this._positive, negative: this._negative }; }
    set models(v: { positive: Model<Props>, negative: Model<Props> }) {
        this._positive = v.positive;
        this._negative = v.negative;
    }
    constructor(op?: {
        defaultScore?: number,
        remainingMsec?: number,
        trimmingScale?: number
    }) {
        this._defaultScore = MlHelper.checkNumericArgument(op?.defaultScore, 0, 1, { includeUpper: true, d: 0 });
        this._remainingMsec = MlHelper.checkNumericArgument(op?.remainingMsec, 0, null, { includeLower: false, d: toMsec(1, TimeUnit.Day) });
        this._trimmingScale = MlHelper.checkNumericArgument(op?.trimmingScale, 1, null, { d: 10000 });
    }
    train(r: { props: Props, isPositive: boolean, timestamp?: number }): void {
        const model = r.isPositive ? this._positive : this._negative;
        Object.entries(r.props).forEach(e => {
            (model as any)[e[0]] ??= {}; // typescript is broken.
            (model[e[0]][e[1].toString()] ??= []).push({ id: ++this._idCounter, timestamp: r.timestamp ?? -1 })
        });
    }
    score(input: Props, thresholdTime: number = Date.now() - this._remainingMsec): number {
        const aggregatePoints = (m: Model<Props>, featureKey: keyof Props, featureValue: string | number) =>
            m[featureKey]?.[featureValue.toString()]
                ?.map(({ timestamp: exp }) => !exp || exp < 0 ? 1 :
                    exp - thresholdTime <= 0 ? 0 : (exp - thresholdTime) / this._remainingMsec)
                ?.reduce((a, b) => a + b) ?? 0;
        const calcP = (k: keyof Props, v: string | number) => {
            const pp = aggregatePoints(this._positive, k, v);
            const np = aggregatePoints(this._negative, k, v);
            return pp === 0 && np === 0 ? null : pp / (pp + np);
        }
        const pset = Object.entries(input).map(e => calcP(e[0], e[1]?.toString()));
        return pset.every(p => p === null) ? this._defaultScore : pset.map(p => p ?? 1).reduce((a, b) => a * b);
    }
    rank(valueSet: Record<keyof Props, (string | number)[]>): [number, Props][] {
        const entries = Object.entries(valueSet), thresholdTime = Date.now() - this._remainingMsec;
        const expand2combinations: (i: number, o?: Props) => Props[] = (i, o = {} as Props) =>
            i === entries.length ? [o] : entries[i][1].flatMap(v => expand2combinations(i + 1, { ...o, [entries[i][0]]: v }));
        return expand2combinations(0).map(p => [this.score(p, thresholdTime), p] as [number, Props]).sort((a, b) => a[0] - b[0]);
    }
    flush(): void {
        const now = Date.now();
        [this._positive, this._negative]
            .flatMap(m => Object.values(m).flatMap((c2e: ValueWeight) => Object.values(c2e)))
            .forEach(exps => UArray.takeOut(exps, ({ id, timestamp: exp }) => exp > 0 && exp < now || id <= this._idCounter - this._trimmingScale));
    }
}
