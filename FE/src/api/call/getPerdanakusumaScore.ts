import axios from "axios"
import { PERDANAKUSUMA_SERVICE_URL } from "../../constants/env"
import { BaseSuccessResponse } from "../contracts/base";
import { GetPerdanakusumaScore} from "../contracts/perdanakusuma";

export default (xRayImage: File) => {
    const formData = new FormData();
    formData.append('imagefile', xRayImage);

    return axios.post<BaseSuccessResponse<GetPerdanakusumaScore>>(`${PERDANAKUSUMA_SERVICE_URL}/predict`, formData)
}