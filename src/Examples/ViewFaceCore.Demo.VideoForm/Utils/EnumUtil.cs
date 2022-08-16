using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ViewFaceCore.Demo.VideoForm.Utils
{
    public class EnumModel
    {
        /// <summary>
        /// 名称
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// 值
        /// </summary>
        public int Value { get; set; }

        /// <summary>
        /// 描述
        /// </summary>
        public string Description { get; set; }
    }

    public class EnumUtil
    {
        /// <summary>
        /// 枚举转对象
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public static List<EnumModel> EnumToList<T>() where T : System.Enum
        {
            List<EnumModel> list = new List<EnumModel>();

            foreach (var e in System.Enum.GetValues(typeof(T)))
            {
                EnumModel m = new EnumModel();
                object[] objArr = e.GetType().GetField(e.ToString()).GetCustomAttributes(typeof(DescriptionAttribute), true);
                if (objArr != null && objArr.Length > 0)
                {
                    object obj = objArr.Where(p => p is DescriptionAttribute).FirstOrDefault();
                    if (obj != null)
                        m.Description = (obj as DescriptionAttribute).Description;
                }
                m.Value = Convert.ToInt32(e);
                m.Name = e.ToString();
                list.Add(m);
            }
            return list;
        }

        public static List<EnumModel> EnumToList(Type t)
        {
            List<EnumModel> list = new List<EnumModel>();

            foreach (var e in System.Enum.GetValues(t))
            {
                EnumModel m = new EnumModel();
                object[] objArr = e.GetType().GetField(e.ToString()).GetCustomAttributes(typeof(DescriptionAttribute), true);
                if (objArr != null && objArr.Length > 0)
                {
                    object obj = objArr.Where(p => p is DescriptionAttribute).FirstOrDefault();
                    if (obj != null)
                        m.Description = (obj as DescriptionAttribute).Description;
                }
                m.Value = Convert.ToInt32(e);
                m.Name = e.ToString();
                list.Add(m);
            }
            return list;
        }
    }
}
