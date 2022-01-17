using System;
using System.Collections.Generic;
using System.Text;

namespace ViewFaceCore.Model
{
    /// <summary>
    /// 眼睛状态结果
    /// </summary>
    public class EyeStateResult
    {
        /// <summary>
        /// 
        /// </summary>
        public EyeStateResult() { }

        /// <summary>
        /// 使用双眼的状态初始化结果
        /// </summary>
        /// <param name="leftEyeState">左眼状态</param>
        /// <param name="rightEyeState">右眼状态</param>
        public EyeStateResult(EyeState leftEyeState, EyeState rightEyeState)
        {
            LeftEyeState = leftEyeState;
            RightEyeState = rightEyeState;
        }

        /// <summary>
        /// 左眼状态
        /// </summary>
        public EyeState LeftEyeState { get; set; }
        /// <summary>
        /// 右眼状态
        /// </summary>
        public EyeState RightEyeState { get; set; }
    }
}
