using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using ViewFaceCore.Demo.VideoForm.Attributes;

namespace ViewFaceCore.Demo.VideoForm.Extensions
{
    /// <summary>
    /// 
    /// </summary>
    public static class MemberInfoExtension
    {
        /// <summary>
        /// 是否忽略
        /// </summary>
        /// <param name="info"></param>
        /// <returns></returns>
        public static bool IsIgnore(this MemberInfo info)
        {
            var attrs = (IsIgnoreAttribute[])info.GetCustomAttributes(typeof(IsIgnoreAttribute), false);
            if (attrs == null || !attrs.Any())
            {
                return false;
            }
            foreach (IsIgnoreAttribute attr in attrs)
            {
                return attr.IsIgnore;
            }
            return false;
        }

        /// <summary>
        /// 是否只读
        /// </summary>
        /// <param name="info"></param>
        /// <returns></returns>
        public static bool IsReadonly(this MemberInfo info)
        {
            var attrs = (IsReadonlyAttribute[])info.GetCustomAttributes(typeof(IsReadonlyAttribute), false);
            if(attrs == null || !attrs.Any())
            {
                return false;
            }
            foreach (IsReadonlyAttribute attr in attrs)
            {
                return attr.IsReadonly;
            }
            return false;
        }

        /// <summary>
        /// 是否隐藏
        /// </summary>
        /// <param name="info"></param>
        /// <returns></returns>
        public static bool IsHidden(this MemberInfo info)
        {
            var attrs = (IsHiddenAttribute[])info.GetCustomAttributes(typeof(IsHiddenAttribute), false);
            if (attrs == null || !attrs.Any())
            {
                return false;
            }
            foreach (IsHiddenAttribute attr in attrs)
            {
                return attr.IsHidden;
            }
            return false;
        }

        /// <summary>
        /// 获取枚举的描述信息
        /// </summary>
        public static string GetDescription(this Enum em)
        {
            Type type = em.GetType();
            FieldInfo fd = type.GetField(em.ToString());
            string des = fd.GetDescription();
            return des;
        }

        /// <summary>
        /// 获取属性的描述信息
        /// </summary>
        public static string GetDescription(this Type type, string proName)
        {
            PropertyInfo pro = type.GetProperty(proName);
            string des = proName;
            if (pro != null)
            {
                des = pro.GetDescription();
            }
            return des;
        }

        /// <summary>
        /// 获取属性的描述信息
        /// </summary>
        public static string GetDescription(this MemberInfo info)
        {
            var attrs = (DescriptionAttribute[])info.GetCustomAttributes(typeof(DescriptionAttribute), false);
            string des = info.Name;
            foreach (DescriptionAttribute attr in attrs)
            {
                des = attr.Description;
            }
            return des;
        }
    }
}
