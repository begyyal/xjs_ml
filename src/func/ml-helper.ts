import { Array2, UType } from "xjs-common";

export namespace MlHelper {
    export function checkNumericArgument(a: number, lower: number, upper: number, op?: {
        d?: number, includeLower?: boolean, includeUpper?: boolean
    }): number {
        const includeLower = op?.includeLower ?? true, includeUpper = !!op?.includeUpper;
        if (UType.isEmpty(a)) return op?.d;
        if (!UType.isNumber(a) || lower && (includeLower ? a < lower : a <= lower) || upper && (includeUpper ? a > upper : a >= upper))
            throw new Error("invalid argument was passed.");
        return a;
    }
    export function calcDistance<Props extends Record<string, number>>(p1: Props, p2: Props): number {
        return Math.sqrt(Array2.sum(Object.keys(p1).map(k => Math.pow(p2[k] - p1[k], 2))));
    }
}