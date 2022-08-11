using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ViewFaceCore.Model;

namespace ViewFaceCore.Configs
{
    public abstract class BaseConfig
    {
        /// <summary>
        /// 识别用的设备类型
        /// </summary>
        /// <remarks>
        /// 目前只能作用CPU，GPU无法使用
        /// </remarks>
        public DeviceType DeviceType { get; set; } = DeviceType.AUTO;

        public Action<string> LogEvent { get; set; }

        /// <summary>
        /// 对外写日志
        /// </summary>
        /// <param name="log"></param>
        internal void WriteLog(string log)
        {
            if (LogEvent != null)
            {
                LogEvent(log);
            }
        }
    }
}
