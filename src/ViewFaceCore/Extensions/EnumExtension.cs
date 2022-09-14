using System;
using System.ComponentModel;
using System.Reflection;

namespace ViewFaceCore.Extensions
{
    /// <summary>
    /// Enum 扩展
    /// </summary>
    internal static class EnumExtension
    {
        /// <summary>
        /// 返回枚举类型的 <see cref="DescriptionAttribute.Description"/>，没有则返回枚举值名称。
        /// </summary>
        /// <param name="enumValue"></param>
        /// <returns></returns>
        public static string ToDescription(this Enum enumValue)
        {
            string value = enumValue.ToString();
            FieldInfo field = enumValue.GetType().GetField(value);
            object[] objs = field.GetCustomAttributes(typeof(DescriptionAttribute), false);    //获取描述属性
            if (objs == null || objs.Length == 0)    //当描述属性没有时，直接返回名称
                return value;
            DescriptionAttribute descriptionAttribute = (DescriptionAttribute)objs[0];
            return descriptionAttribute.Description;
        }
    }
}
