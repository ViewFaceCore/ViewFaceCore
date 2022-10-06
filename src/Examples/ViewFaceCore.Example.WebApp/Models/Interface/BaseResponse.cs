namespace ViewFaceCore.Example.WebApp.Models.Interface
{
    public class BaseResponse<T> : IResponse<T> where T : IResponseData
    {
        public BaseResponse(T data)
        {
            Code = 0;
            Message = "Success";
            Data = data;
        }

        public BaseResponse(int code, string message)
        {
            Code = code;
            Message = message;
            Data = default;
        }

        public BaseResponse(int code, string message, T data)
        {
            Code = code;
            Message = message;
            Data = data;
        }

        public int Code { get; set; }

        public string Message { get; set; }

        public T Data { get; set; }
    }
}
