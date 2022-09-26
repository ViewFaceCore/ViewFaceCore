using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using ViewFaceCore.Demo.VideoForm.Utils;

namespace ViewFaceCore.Demo.VideoForm.Extensions
{
    public static class DataGridViewExtension
    {
        public static void Bind<T>(this DataGridView dataGrid, List<T> entityList, Dictionary<DataGridViewColumn, object> extraCols = null)
        {
            //取出第一个实体的所有Propertie
            Type entityType = typeof(T);
            PropertyInfo[] entityProperties = entityType.GetProperties();
            //添加了的列
            List<PropertyInfo> properties = InitColumns<T>(dataGrid, extraCols);
            if (!properties.Any())
            {
                return;
            }
            if (entityList == null || !entityList.Any())
            {
                return;
            }
            foreach (object entity in entityList)
            {
                //检查所有的的实体都为同一类型
                if (entity.GetType() != entityType)
                {
                    throw new Exception("要转换的集合元素类型不一致");
                }
                object[] entityValues = new object[properties.Count + extraCols.Count];
                for (int i = 0; i < properties.Count; i++)
                {
                    object value = properties[i].GetValue(entity, null);
                    if (entityProperties[i].PropertyType == typeof(DateTime))
                    {
                        DateTime? dateTime = (DateTime?)value;
                        if(dateTime != null && dateTime != DateTime.MinValue)
                        {
                            entityValues[i] = dateTime.Value.ToString("yyyy/MM/dd HH:mm:ss");
                        }
                        else
                        {
                            entityValues[i] = null;
                        }
                    }
                    else if (entityProperties[i].PropertyType?.BaseType == typeof(Enum))
                    {
                        entityValues[i] = Convert.ToInt32(value);
                    }
                    else
                    {
                        entityValues[i] = value?.ToString();
                    }
                }
                if (extraCols != null && extraCols.Any())
                {
                    int lastCount = properties.Count;
                    object[] extraValues = extraCols.Values.ToArray();
                    for (int i = 0; i < extraValues.Length; i++)
                    {
                        entityValues[i + lastCount] = extraValues[i];
                    }
                }
                dataGrid.Rows.Add(entityValues);
            }
        }

        private static List<PropertyInfo> InitColumns<T>(DataGridView dataGrid, Dictionary<DataGridViewColumn, object> extraCols = null)
        {
            //取出第一个实体的所有Propertie
            Type entityType = typeof(T);
            PropertyInfo[] entityProperties = entityType.GetProperties();
            List<PropertyInfo> properties = new List<PropertyInfo>();
            for (int i = 0; i < entityProperties.Length; i++)
            {
                if (entityProperties[i].IsIgnore())
                {
                    continue;
                }
                string desc = entityProperties[i].GetDescription() ?? entityProperties[i].Name;
                bool isReadonly = entityProperties[i].IsReadonly();
                bool isHidden = entityProperties[i].IsHidden();
                switch (entityProperties[i].PropertyType?.BaseType.FullName)
                {
                    case "System.ValueType":
                        {
                            switch (entityProperties[i].PropertyType.FullName)
                            {
                                case "System.Boolean":
                                    dataGrid.Columns.Add(new DataGridViewCheckBoxColumn()
                                    {
                                        Name = entityProperties[i].Name,
                                        HeaderText = desc,
                                        ReadOnly = isReadonly,
                                        Visible = !isHidden,
                                    });
                                    break;
                                default:
                                    dataGrid.Columns.Add(new DataGridViewTextBoxColumn()
                                    {
                                        Name = entityProperties[i].Name,
                                        HeaderText = desc,
                                        ReadOnly = isReadonly,
                                        Visible = !isHidden,
                                    });
                                    break;
                            }
                            break;
                        }
                    case "System.Object":
                        {
                            dataGrid.Columns.Add(new DataGridViewTextBoxColumn()
                            {
                                Name = entityProperties[i].Name,
                                HeaderText = desc,
                                ReadOnly = isReadonly,
                                Visible = !isHidden,
                            });
                            break;
                        }
                    case "System.Enum":
                        {
                            var col = new DataGridViewComboBoxColumn()
                            {
                                Name = entityProperties[i].Name,
                                HeaderText = desc,
                                ReadOnly = isReadonly,
                                Visible = !isHidden,
                            };
                            col.DataSource = EnumUtil.EnumToList(entityProperties[i].PropertyType);
                            col.DisplayMember = nameof(EnumModel.Description);
                            col.ValueMember = nameof(EnumModel.Value);
                            dataGrid.Columns.Add(col);
                            break;
                        }
                    default:
                        break;
                }
                properties.Add(entityProperties[i]);
            }
            if (extraCols != null)
            {
                foreach (var item in extraCols)
                {
                    dataGrid.Columns.Add(item.Key);
                }
            }
            return properties;
        }

        private static bool IsNullable(object obj)
        {
            var type = obj.GetType();
            return type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Nullable<>);
        }
    }
}
