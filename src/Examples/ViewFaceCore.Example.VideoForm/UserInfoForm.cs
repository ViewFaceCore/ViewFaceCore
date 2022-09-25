using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using ViewFaceCore.Core;
using ViewFaceCore.Demo.VideoForm.Extensions;
using ViewFaceCore.Demo.VideoForm.Models;
using ViewFaceCore.Demo.VideoForm.Utils;
using ViewFaceCore.Extensions;
using ViewFaceCore.Models;

namespace ViewFaceCore.Demo.VideoForm
{
    public partial class UserInfoForm : Form
    {
        private readonly UserInfoFormParam _takePhotoInfo = null;
        private UserInfo _globalUserInfo = new UserInfo();

        public UserInfoForm(UserInfoFormParam takePhotoInfo)
        {
            _takePhotoInfo = takePhotoInfo;
            InitializeComponent();
        }

        private void SaveUserForm_Load(object sender, EventArgs e)
        {
            if (_takePhotoInfo?.Bitmap == null)
            {
                MessageBox.Show("未获取到拍摄的图片，请重试！", "警告", MessageBoxButtons.OK, MessageBoxIcon.Error);
                this.Close();
                return;
            }
            //绑定数据源
            List<EnumModel> genderEnums = EnumUtil.EnumToList<GenderEnum>().OrderByDescending(p => p.Value).ToList();
            comboBoxGender.DataSource = genderEnums;
            comboBoxGender.DisplayMember = nameof(EnumModel.Description);
            comboBoxGender.ValueMember = nameof(EnumModel.Value);
            comboBoxGender.SelectedIndex = -1;

            if (_takePhotoInfo.ReadOnly)
            {
                //查看模式
                this.Text = $"查看【{_takePhotoInfo.UserInfo.Name}】信息";

                if (_takePhotoInfo?.UserInfo == null)
                {
                    MessageBox.Show("未获取到人员信息，请重试！", "警告", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    this.Close();
                    return;
                }

                SetUIStatus(false);
                this.btnClear.Visible = false;
                this.btnSave.Visible = false;

                FormHelper.SetTextBoxText(this.tbName, _takePhotoInfo.UserInfo.Name);
                FormHelper.SetTextBoxText(this.tbAge, _takePhotoInfo.UserInfo.Age.ToString());
                FormHelper.SetTextBoxText(this.tbPhone, _takePhotoInfo.UserInfo.Phone);
                FormHelper.SetTextBoxText(this.tbRemark, _takePhotoInfo.UserInfo.Remark);
                SetGenderComboBoxValue(this.comboBoxGender, (int)_takePhotoInfo.UserInfo.Gender);
                this.pictureBoxUser.Image = _takePhotoInfo.Bitmap;
            }
            else
            {
                //编辑模式
                this.Text = "保存用户信息";
                this.pictureBoxUser.Image = _takePhotoInfo.Bitmap;
                //识别人脸
                FaceDetector(_takePhotoInfo.Bitmap.DeepClone());
            }
        }

        private void DrawingFaceInfo(Bitmap bitmap, List<Models.VideoFaceInfo> faceInfos)
        {
            if (!faceInfos.Any())
            {
                return;
            }
            using (Graphics g = Graphics.FromImage(bitmap))
            {
                g.DrawRectangles(new Pen(Color.Red, 4), faceInfos.Select(p => p.Rectangle).ToArray());
            }
        }

        private void btnSave_Click(object sender, EventArgs e)
        {
            try
            {
                SetUIStatus(false);
                UserInfo userInfo = BuildUserInfo();
                if (userInfo == null)
                {
                    throw new Exception("获取用户基本信息失败！");
                }
                using (DefaultDbContext db = new DefaultDbContext())
                {
                    db.UserInfo.Add(userInfo);
                    if (db.SaveChanges() > 0)
                    {
                        CacheManager.Instance.Refesh();
                        this.Close();
                        _ = Task.Run(() =>
                        {
                            //确保关闭后弹窗
                            Thread.Sleep(100);
                            MessageBox.Show("保存用户信息成功！", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);

                        });
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "警告", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            finally
            {
                SetUIStatus(false);
            }
        }

        private void btnClear_Click(object sender, EventArgs e)
        {
            this.tbName.Text = "";
            this.tbAge.Text = "";
            this.tbPhone.Text = "";
            this.tbRemark.Text = "";
            this.comboBoxGender.SelectedIndex = -1;
        }

        private void tbAge_KeyPress(object sender, KeyPressEventArgs e)
        {
            if (e.KeyChar != '\b' && !Char.IsDigit(e.KeyChar))
            {
                e.Handled = true;
            }
        }

        private void FaceDetector(Bitmap bitmap)
        {
            _ = Task.Run((Action)(() =>
            {
                ViewFaceFactory faceFactory = new ViewFaceFactory(bitmap.Width, bitmap.Height);
                try
                {
                    SetUIStatus(false);
                    FormHelper.SetLabelStatus(labelStatus, "人脸检测中...", true);

                    using (FaceImage faceImage = bitmap.ToFaceImage())
                    {
                        FaceInfo[] faceInfos = faceFactory.Get<FaceDetector>().Detect((FaceImage)faceImage);
                        if (faceInfos == null || !faceInfos.Any())
                        {
                            throw new Exception("未获取到人脸信息！");
                        }
                        FaceMarkPoint[] markPoints = Core.Extensions.Mark(faceFactory.Get<FaceLandmarker>(), bitmap, faceInfos[0]);
                        if (markPoints == null)
                        {
                            throw new Exception("检测人脸信息失败：标记人脸失败！");
                        }
                        MaskDetector maskDetector = faceFactory.Get<MaskDetector>();
                        PlotMaskResult maskResult = Core.Extensions.Detect(maskDetector, bitmap, faceInfos[0]);
                        if (maskResult.Masked)
                        {
                            throw new Exception("人脸不能有任何遮挡或者戴有口罩！");
                        }
                        AgePredictor agePredictor = faceFactory.Get<AgePredictor>();
                        _globalUserInfo.Age = agePredictor.PredictAgeWithCrop((FaceImage)faceImage, markPoints);

                        GenderPredictor genderPredictor = faceFactory.Get<GenderPredictor>();
                        Gender gender = genderPredictor.PredictGenderWithCrop((FaceImage)faceImage, markPoints);
                        switch (gender)
                        {
                            case Gender.Male:
                                _globalUserInfo.Gender = GenderEnum.Male;
                                break;
                            case Gender.Female:
                                _globalUserInfo.Gender = GenderEnum.Female;
                                break;
                            case Gender.Unknown:
                                _globalUserInfo.Gender = GenderEnum.Unknown;
                                break;
                        }

                        FaceRecognizer faceRecognizer = faceFactory.Get<FaceRecognizer>();
                        float[] extractData = faceRecognizer.Extract((FaceImage)faceImage, markPoints);
                        if (extractData == null)
                        {
                            throw new Exception("识别人脸信息失败！");
                        }
                        _globalUserInfo.Extract = string.Join(";", extractData);

                        #region 设置默认输入框值

                        FormHelper.SetTextBoxText(tbName, _globalUserInfo.Name);
                        FormHelper.SetTextBoxText(tbAge, _globalUserInfo.Age.ToString());
                        SetGenderComboBoxValue(comboBoxGender, (int)_globalUserInfo.Gender);

                        //画方框
                        DrawingFaceInfo(bitmap, new List<Models.VideoFaceInfo>(){
                            new Models.VideoFaceInfo()
                            {
                                Location = faceInfos[0].Location,
                            }
                        });
                        FormHelper.SetPictureBoxImage(this.pictureBoxUser, bitmap.DeepClone());
                        #endregion
                    }
                    _globalUserInfo.Image = Base64.BitmapToString(bitmap);
                    FormHelper.SetLabelStatus(labelStatus, "人脸检测完成！", true);
                    SetUIStatus(true);
                }
                catch (Exception ex)
                {
                    FormHelper.SetLabelStatus(labelStatus, ex.Message, true);
                    MessageBox.Show(ex.Message, "警告", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                finally
                {
                    faceFactory?.Dispose();
                    bitmap?.Dispose();
                }
            }));
        }

        private void SetUIStatus(bool status)
        {
            FormHelper.SetControlStatus(this.tbName, status);
            FormHelper.SetControlStatus(this.tbAge, status);
            FormHelper.SetControlStatus(this.comboBoxGender, status);
            FormHelper.SetControlStatus(this.tbPhone, status);
            FormHelper.SetControlStatus(this.tbRemark, status);
            FormHelper.SetControlStatus(this.btnClear, status);
            FormHelper.SetControlStatus(this.btnSave, status);
        }

        private void SetGenderComboBoxValue(ComboBox comboBox, int value)
        {
            Action action = new Action(() =>
            {
                if (comboBoxGender.DataSource == null || !(comboBoxGender.DataSource is List<EnumModel>))
                {
                    return;
                }
                var source = comboBoxGender.DataSource as List<EnumModel>;
                var selectItem = source.Where(p => p.Value == value).FirstOrDefault();
                if (selectItem == null)
                    return;
                comboBoxGender.SelectedValue = selectItem.Value;
            });

            if (comboBox.InvokeRequired)
            {
                comboBox.Invoke(action);
            }
            else
            {
                action();
            }
        }

        private UserInfo BuildUserInfo()
        {
            string name = tbName.Text;
            if (string.IsNullOrEmpty(name))
            {
                throw new Exception("用户名不能为空！");
            }
            if (tbPhone.Text.Length > 15)
            {
                throw new Exception("用户名不能超过15个字符！");
            }
            int age = 0;
            if (!string.IsNullOrEmpty(tbAge.Text))
            {
                if (!int.TryParse(tbAge.Text, out age) || age > 300)
                {
                    throw new Exception("年龄格式错误！");
                }
            }
            if (tbPhone.Text.Length > 20)
            {
                throw new Exception("电话号码错误！");
            }
            if (tbRemark.Text.Length > 150)
            {
                throw new Exception("备注字数不能超过150个字！");
            }
            GenderEnum gender = GenderEnum.Unknown;
            if (!string.IsNullOrEmpty(comboBoxGender.Text)
                && comboBoxGender.SelectedItem != null
                && comboBoxGender.SelectedItem is EnumModel)
            {
                EnumModel model = comboBoxGender.SelectedItem as EnumModel;
                gender = (GenderEnum)(model.Value);
            }
            if (string.IsNullOrWhiteSpace(_globalUserInfo.Image))
            {
                throw new Exception("图片不能为空！");
            }
            return new UserInfo()
            {
                Name = name,
                Age = age,
                Gender = gender,
                Remark = tbRemark.Text,
                Phone = tbPhone.Text,
                Image = _globalUserInfo.Image,
                Extract = _globalUserInfo.Extract,
                IsDelete = false,
                CreateTime = DateTime.Now,
                UpdateTime = null,
            };
        }

        private void SaveUserForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            try
            {
                if (_takePhotoInfo != null && _takePhotoInfo.Bitmap != null)
                {
                    _takePhotoInfo.Bitmap.Dispose();
                }
            }
            catch { }
        }
    }
}
