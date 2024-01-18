interface BaseResponse {
    code: number;
    message: string;
    success: boolean;
}

export interface BaseSuccessResponse<R> extends BaseResponse {
    data: R;
}

interface ErrorRequest {
    field: string;
    error: string;
}

export interface BaseErrorResponse extends BaseResponse {
    errors: ErrorRequest[];
}