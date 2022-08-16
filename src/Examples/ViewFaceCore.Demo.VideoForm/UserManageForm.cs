using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using ViewFaceCore.Demo.VideoForm.Extensions;
using ViewFaceCore.Demo.VideoForm.Models;
using ViewFaceCore.Demo.VideoForm.Utils;

namespace ViewFaceCore.Demo.VideoForm
{
    public partial class UserManageForm : Form
    {
        public UserManageForm()
        {
            InitializeComponent();
        }

        private void 保存更改ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            for (int i = 0; i < userDataGrid.Rows.Count; i++)
            {
                for (int j = 0; j < userDataGrid.Rows[i].Cells.Count; j++)
                {
                    if (userDataGrid.Rows[i].Cells[j].IsInEditMode)
                    {
                        userDataGrid.CurrentCell = userDataGrid[0, 0];
                        userDataGrid.CurrentCell = userDataGrid[userDataGrid.Rows[i].Cells[j].ColumnIndex, userDataGrid.Rows[i].Cells[j].RowIndex];
                    }
                }
            }
            Dictionary<int, UserInfo> userInfos = RowToUserInfo();
            if (userInfos == null || !userInfos.Any())
            {
                MessageBox.Show("没有用户记录！", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            using (var db = new DefaultDbContext())
            {
                List<int> ids = userInfos.Values.Select(x => x.Id).ToList();
                List<UserInfo> attUsers = db.UserInfo.Where(p => ids.Contains(p.Id) && !p.IsDelete).ToList();
                Dictionary<int, UserInfo> updateInfos = new Dictionary<int, UserInfo>();
                foreach (var item in attUsers)
                {
                    var modUser = userInfos.Where(p => p.Value.Id == item.Id).FirstOrDefault();
                    if (modUser.Value == null) continue;

                    bool isUpdate = false;
                    if (item.Name != modUser.Value.Name)
                    {
                        item.Name = modUser.Value.Name;
                        isUpdate = true;
                    }
                    if (item.Age != modUser.Value.Age)
                    {
                        item.Age = modUser.Value.Age;
                        isUpdate = true;
                    }
                    if (item.Gender != modUser.Value.Gender)
                    {
                        item.Gender = modUser.Value.Gender;
                        isUpdate = true;
                    }
                    if (item.Remark != modUser.Value.Remark)
                    {
                        item.Remark = modUser.Value.Remark;
                        isUpdate = true;
                    }
                    if (item.Phone != modUser.Value.Phone)
                    {
                        item.Phone = modUser.Value.Phone;
                        isUpdate = true;
                    }
                    if (isUpdate)
                    {
                        item.UpdateTime = DateTime.Now;
                        updateInfos.Add(modUser.Key, item);
                    }
                }
                if (updateInfos.Any())
                {
                    db.UserInfo.UpdateRange(updateInfos.Values);
                    if (db.SaveChanges() == updateInfos.Count)
                    {
                        foreach (var item in updateInfos)
                        {
                            UpdateRow(item.Key, item.Value);
                        }
                        MessageBox.Show("保存成功！", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                    else
                    {
                        MessageBox.Show("保存失败，请重试！", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                }
                else
                {
                    MessageBox.Show("没有任何变更记录！", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
        }

        private void UserManageForm_Load(object sender, EventArgs e)
        {
            Fill();
        }

        private async void Fill()
        {
            using (var db = new DefaultDbContext())
            {
                List<UserInfo> userInfos = await db.UserInfo.Where(p => !p.IsDelete).ToListAsync();
                if (userInfos == null)
                {
                    userInfos = new List<UserInfo>();
                }
                var buttonColumn = new DataGridViewButtonColumn()
                {
                    Name = "Ext_Options",
                    HeaderText = "操作",
                    UseColumnTextForButtonValue = false,
                };
                Dictionary<DataGridViewColumn, object> extraCols = new Dictionary<DataGridViewColumn, object>()
                {
                    {new DataGridViewButtonColumn(),"查看" },
                    {new DataGridViewButtonColumn(),"删除" },
                };
                userDataGrid.Bind(userInfos, extraCols);
            }
        }

        private void userDataGrid_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {
            if (userDataGrid.Columns[e.ColumnIndex] is DataGridViewButtonColumn)
            {
                DataGridViewButtonColumn btnCol = userDataGrid.Columns[e.ColumnIndex] as DataGridViewButtonColumn;
                if (btnCol == null)
                {
                    MessageBox.Show("当前行内容为空！", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                }
                DataGridViewRow row = userDataGrid.Rows[e.RowIndex];
                if (row == null)
                {
                    MessageBox.Show("当前行内容为空！", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                }
                UserInfo userInfo = RowToUserInfo(row);
                if (userInfo == null || userInfo.Id < 0)
                {
                    MessageBox.Show("当前行内容为空！", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                }
                switch (row.Cells[e.ColumnIndex].Value)
                {
                    case "删除":
                        {
                            if (MessageBox.Show($"确认删除用户【{userInfo.Name}】吗？", "提示", MessageBoxButtons.OKCancel, MessageBoxIcon.Warning) == DialogResult.OK)
                            {
                                using (var db = new DefaultDbContext())
                                {
                                    int rows = db.UserInfo.Where(p => p.Id == userInfo.Id).ToDelete().ExecuteAffrows();
                                    if (rows > 0)
                                    {
                                        userDataGrid.Rows.RemoveAt(e.RowIndex);
                                    }
                                }
                            }
                        }
                        break;
                    case "查看":
                        {
                            UserInfoForm userInfoForm = new UserInfoForm(new TakePhotoInfo()
                            {
                                UserInfo = userInfo,
                                Bitmap = Base64.StringToImage(userInfo.Image),
                                ReadOnly = true,
                            });
                            userInfoForm.Show();
                        }
                        break;
                    default:
                        MessageBox.Show($"未知的操作类型：{row.Cells[e.ColumnIndex].Value}", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        break;
                }

                
                
            }
        }

        private void 刷新ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            userDataGrid.Rows.Clear();
            userDataGrid.Columns.Clear();
            Fill();
        }

        #region 数据转换

        private UserInfo RowToUserInfo(DataGridView userDataGrid, int index)
        {
            if (userDataGrid != null && index >= 0)
            {
                return RowToUserInfo(userDataGrid.Rows[index]);
            }
            return null;
        }

        private UserInfo RowToUserInfo(DataGridViewRow row)
        {
            if (row == null) return null;
            UserInfo userInfo = new UserInfo()
            {
                Id = int.TryParse(row.Cells[nameof(UserInfo.Id)]?.Value?.ToString(), out int id) ? id : -1,
                Name = row.Cells[nameof(UserInfo.Name)]?.Value?.ToString(),
                Age = int.TryParse(row.Cells[nameof(UserInfo.Age)]?.Value?.ToString(), out int age) ? age : 0,
                Gender = int.TryParse(row.Cells[nameof(UserInfo.Gender)]?.Value?.ToString(), out int genderInt) ? (GenderEnum)genderInt : GenderEnum.Unknown,
                Remark = row.Cells[nameof(UserInfo.Remark)]?.Value?.ToString(),
                Phone = row.Cells[nameof(UserInfo.Phone)]?.Value?.ToString(),
                Image = row.Cells[nameof(UserInfo.Image)]?.Value?.ToString(),
                Extract = row.Cells[nameof(UserInfo.Extract)]?.Value?.ToString(),
                CreateTime = DateTime.TryParse(row.Cells[nameof(UserInfo.CreateTime)]?.Value?.ToString(), out DateTime createTime) ? createTime : DateTime.MinValue,
                UpdateTime = DateTime.TryParse(row.Cells[nameof(UserInfo.UpdateTime)]?.Value?.ToString(), out DateTime updateTime) ? updateTime : DateTime.MinValue,
            };
            return userInfo;
        }

        private Dictionary<int, UserInfo> RowToUserInfo()
        {
            Dictionary<int, UserInfo> userInfos = new Dictionary<int, UserInfo>();
            foreach (DataGridViewRow row in userDataGrid.Rows)
            {
                UserInfo userInfo = RowToUserInfo(row);
                if (userInfo != null && userInfo.Id >= 0)
                {
                    userInfos.Add(row.Index, userInfo);
                }
            }
            return userInfos;
        }

        private void UpdateRow(int index, UserInfo userInfo)
        {
            if (userDataGrid.Rows.Count <= index)
            {
                return;
            }
            userDataGrid.Rows[index].Cells[nameof(UserInfo.UpdateTime)].Value = userInfo.UpdateTime?.ToString("yyyy-MM-dd HH:mm:ss");
        }

        #endregion
    }
}
