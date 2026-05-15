import { Array2, UType } from "xjs-common";
import { MlHelper } from "../func/ml-helper";

interface CommonOption<C, P> {
    trimmingScale?: number;
    modifiedRatio?: number;
    coef?: Partial<Record<keyof P, number>>;
    defaultCls?: C;
}
export class KnnClassifier<Cls extends string, Props extends Record<string, number>> {
    private _dataSet: [Cls, Props][] = [];
    get dataSet() { return this._dataSet; }
    set dataSet(v: [Cls, Props][]) {
        this._dataSet = [...v];
        if (this._dataSet.length > this._trimmingScale) this.dataSet.splice(this._trimmingScale);
    }
    private readonly _staticK: number;
    private readonly _minK: number;
    private readonly _kRatio: number;
    private readonly _trimmingScale: number;
    private readonly _modifiedRatio: number;
    private readonly _prop2coef?: Partial<Record<string, number>>;
    private readonly _defaultCls?: Cls;
    private get defaultCls() {
        if (this._defaultCls) return this._defaultCls;
        throw Error("no default class was set.");
    }
    private get k() {
        if (this._staticK) return this._staticK;
        return this._dataSet.length <= this._minK / this._kRatio ? this._minK : Math.ceil(this._dataSet.length * this._kRatio);
    }
    constructor(op?: CommonOption<Cls, Props> & { k?: number });
    constructor(op?: CommonOption<Cls, Props> & { kRatio?: number, minK?: number });
    constructor(op?: CommonOption<Cls, Props> & { k?: number, kRatio?: number, minK?: number }) {
        this._staticK = op?.k;
        this._minK = MlHelper.checkNumericArgument(op?.minK, 1, null, { d: 5 });
        this._kRatio = MlHelper.checkNumericArgument(op?.kRatio, 0, 1, { includeLower: false, includeUpper: true, d: 0.05 });
        this._trimmingScale = MlHelper.checkNumericArgument(op?.trimmingScale, 1, null, { d: 10000 });
        this._modifiedRatio = MlHelper.checkNumericArgument(op?.modifiedRatio, 0, 1, { d: 0.5, includeUpper: true });
        this._prop2coef = op?.coef && Object.keys(op.coef).length > 0 && { ...op.coef };
        if (this._prop2coef) Object.keys(this._prop2coef)
            .filter(k => !UType.isNumber(this._prop2coef[k]))
            .forEach(k => delete this._prop2coef[k]);
        this._defaultCls = op?.defaultCls;
    }
    setData(r: { props: Props, cls: Cls }): void {
        this._dataSet.push([r.cls, r.props]);
        if (this._dataSet.length > this._trimmingScale) this._dataSet.shift();
    }
    classify(props: Props): Cls {
        if (this._dataSet.length === 0) return this.defaultCls;
        const propsModified = this.applyCoef2props(props);
        const dataInK = this._dataSet.map(tpl => [tpl[0], MlHelper.calcDistance(propsModified, this.applyCoef2props(tpl[1]))] as [Cls, number])
            .sort((a, b) => a[1] - b[1]).slice(0, this.k);
        const farthest = dataInK.at(-1)[1];
        return Array.from(Array2.map(dataInK, d => d[0]).entries())
            .map(e => [e[0], Array2.sum(e[1].map(tpl => 1 - this._modifiedRatio * tpl[1] / farthest))] as [Cls, number])
            .sort((a, b) => a[1] - b[1]).at(-1)[0];
    }
    private applyCoef2props(props: Props): Props {
        const propsModified = { ...props } as any; // typescript is broken.
        if (this._prop2coef) Object.entries(this._prop2coef).forEach(e => propsModified[e[0]] = propsModified[e[0]] * e[1]);
        return propsModified;
    }
}