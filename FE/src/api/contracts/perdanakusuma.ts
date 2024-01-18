import { PerdanakusumaScore } from "../../models/perdanakusuma";

export interface GetPerdanakusumaScore {
    predictions : PerdanakusumaScore;
    status : string;
    total : number;
    severity: string;
}