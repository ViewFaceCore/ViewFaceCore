using AForge.Video.DirectShow;
using System;
using System.Collections.Generic;
using System.Diagnostics;
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
using ViewFaceCore.Models;

namespace ViewFaceCore.Demo.VideoForm
{
    public partial class MainForm : Form
    {
        private const string _defaultCapability = "1280x720";
        private const string _enableBtnString = "关闭摄像头";
        private const string _disableBtnString = "打开摄像头并识别人脸";

        private bool isDetecting = false;

        /// <summary>
        /// 摄像头设备信息集合
        /// </summary>
        private FilterInfoCollection videoDevices;

        /// <summary>
        /// 取消令牌
        /// </summary>
        private CancellationTokenSource token = null;

        private ViewFaceFactory faceFactory = null;

        public MainForm()
        {
            InitializeComponent();
        }

        #region Events

        /// <summary>
        /// 窗体加载时
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Form_Load(object sender, EventArgs e)
        {
            // 隐藏摄像头画面控件
            VideoPlayer.Visible = false;
            //初始化VideoDevices
            检测摄像头ToolStripMenuItem_Click(null, null);
            //默认禁用拍照按钮
            FormHelper.SetControlStatus(this.ButtonSave, false);
        }

        /// <summary>
        /// 窗体关闭时，关闭摄像头
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Form_Closing(object sender, FormClosingEventArgs e)
        {
            Stop();
        }

        /// <summary>
        /// 点击开始按钮时
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void ButtonStart_Click(object sender, EventArgs e)
        {
            if (ButtonStart.Text == _disableBtnString)
            {
                //开始
                Start();
            }
            else if (ButtonStart.Text == _enableBtnString)
            {
                //停止
                Stop();
            }
            else
            {
                MessageBox.Show($"Emmmmm...姿势不对？？？", "警告", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            var videoCapture = new VideoCaptureDevice(videoDevices[comboBox1.SelectedIndex].MonikerString);
            List<string> supports = videoCapture.VideoCapabilities.OrderBy(p => p.FrameSize.Width).Select(p => $"{p.FrameSize.Width}x{p.FrameSize.Height}").ToList();
            this.comboBox2.DataSource = supports;
            if (supports.Contains(_defaultCapability))
            {
                comboBox2.SelectedIndex = supports.IndexOf(_defaultCapability);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void CheckBoxDetect_CheckedChanged(object sender, EventArgs e)
        {
            CheckBoxFaceProperty.Enabled = CheckBoxDetect.Checked;
            CheckBoxFaceMask.Enabled = CheckBoxDetect.Checked;
            CheckBoxFPS.Enabled = CheckBoxDetect.Checked;
            numericUpDownFPSTime.Enabled = CheckBoxDetect.Checked;
        }

        private void 人员管理ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            UserManageForm userManageForm = new UserManageForm();
            userManageForm.Show();
        }

        private void 退出ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        private void 关于ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            MessageBox.Show("乌拉~", "关于", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void ButtonSave_Click(object sender, EventArgs e)
        {
            if (!VideoPlayer.IsRunning)
            {
                MessageBox.Show("请先开启人脸识别！", "警告", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            FormHelper.SetControlStatus(ButtonSave, false);
            _ = Task.Run(() =>
            {
                using (Bitmap bitmap = VideoPlayer.GetCurrentVideoFrame())
                {
                    if (bitmap == null)
                    {
                        MessageBox.Show("拍照失败，没有获取到图像，请重试！", "警告", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        FormHelper.SetControlStatus(ButtonSave, true);
                        return;
                    }
                    else
                    {
                        UserInfoFormParam formParam = new UserInfoFormParam()
                        {
                            Bitmap = bitmap.DeepClone(),
                        };
                        FormHelper.SetControlStatus(ButtonSave, true);
                        //打开保存框
                        UserInfoForm saveUser = new UserInfoForm(formParam);
                        saveUser.ShowDialog();
                    }
                }
            });
        }

        private void 检测摄像头ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (this.comboBox1.Enabled)
            {
                videoDevices = new FilterInfoCollection(FilterCategory.VideoInputDevice);
                comboBox1.Items.Clear();
                comboBox2.DataSource = null;
                foreach (FilterInfo info in videoDevices)
                {
                    comboBox1.Items.Add(info.Name);
                }
                if (comboBox1.Items.Count > 0)
                {
                    comboBox1.SelectedIndex = 0;
                }
            }
        }

        private void 强制刷新缓存ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            CacheManager.Instance.Refesh();
            MessageBox.Show($"缓存已刷新！", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        #endregion

        #region Methods

        private (int width, int height) GetSelectCapability()
        {
            string selectStr = this.comboBox2.SelectedItem.ToString();
            if (string.IsNullOrEmpty(selectStr))
            {
                selectStr = _defaultCapability;
            }
            string[] items = selectStr.Split('x');
            if (items.Length != 2)
            {
                throw new Exception("Get capability from select item failed.");
            }
            return (int.Parse(items[0]), int.Parse(items[1]));
        }

        private void Start()
        {
            if (VideoPlayer.IsRunning)
            {
                Stop();
            }

            FormHelper.SetControlStatus(this.comboBox1, false);
            FormHelper.SetControlStatus(this.comboBox2, false);
            FormHelper.SetControlStatus(this.ButtonStart, false);
            FormHelper.SetControlStatus(this.ButtonSave, false);
            bool isSuccess = true;

            try
            {
                if (comboBox1.SelectedIndex < 0)
                {
                    MessageBox.Show($"没有找到可用的摄像头，请重试！", "警告", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    isSuccess = false;
                    return;
                }
                (int width, int height) = GetSelectCapability();
                if (faceFactory == null)
                {
                    faceFactory = new ViewFaceFactory(width, height);
                }
                FilterInfo info = videoDevices[comboBox1.SelectedIndex];
                VideoCaptureDevice videoCapture = new VideoCaptureDevice(info.MonikerString);
                var videoResolution = videoCapture.VideoCapabilities.Where(p => p.FrameSize.Width == width && p.FrameSize.Height == height).FirstOrDefault();
                if (videoResolution == null)
                {
                    List<string> supports = videoCapture.VideoCapabilities.OrderBy(p => p.FrameSize.Width).Select(p => $"{p.FrameSize.Width}x{p.FrameSize.Height}").ToList();
                    string supportStr = "无，或获取失败";
                    if (supports.Any())
                    {
                        supportStr = string.Join("|", supports);
                    }
                    MessageBox.Show($"摄像头不支持拍摄分辨率为{width}x{height}的视频，请重新指定分辨率。\n支持分辨率：{supportStr}", "警告", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    isSuccess = false;
                    return;
                }
                videoCapture.VideoResolution = videoResolution;
                VideoPlayer.VideoSource = videoCapture;
                VideoPlayer.Start();

                Stopwatch stopwatch = Stopwatch.StartNew();
                while (!VideoPlayer.IsRunning)
                {
                    if (stopwatch.ElapsedMilliseconds > 10000)
                    {
                        //10s超时
                        stopwatch.Stop();
                        isSuccess = false;
                        MessageBox.Show($"开启失败，请重试！", "警告", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        return;
                    }
                    Thread.Sleep(1);
                }
                stopwatch.Stop();
                CacheManager.Instance.Refesh();
                token = new CancellationTokenSource();
                StartDetector(token.Token);
            }
            finally
            {
                if (isSuccess)
                {
                    FormHelper.SetButtonText(ButtonStart, _enableBtnString);
                    FormHelper.SetControlStatus(this.ButtonStart, true);
                    FormHelper.SetControlStatus(this.ButtonSave, true);
                }
                else
                {
                    FormHelper.SetControlStatus(this.comboBox1, true);
                    FormHelper.SetControlStatus(this.comboBox2, true);
                    FormHelper.SetControlStatus(this.ButtonStart, true);
                    FormHelper.SetControlStatus(this.ButtonSave, false);
                }
            }
        }

        private async void Stop()
        {
            try
            {
                if (!VideoPlayer.IsRunning)
                {
                    return;
                }
                FormHelper.SetControlStatus(this.ButtonStart, false);
                VideoPlayer?.SignalToStop();
                VideoPlayer?.WaitForStop();
                token?.Cancel();

                FormHelper.SetButtonText(ButtonStart, "关闭中...");
                bool isStopped = true;
                //等待处理完数据
                await Task.Run(() =>
                {
                    Stopwatch stopwatch = Stopwatch.StartNew();
                    while (isDetecting)
                    {
                        if (stopwatch.ElapsedMilliseconds > 10000)
                        {
                            isStopped = false;
                            break;
                        }
                    }
                    stopwatch.Stop();
                });
                if (!isStopped)
                {
                    MessageBox.Show($"错误：关闭摄像头超时！", "警告", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                //设置图形控件为空白
                FormHelper.SetPictureBoxImage(FacePictureBox, null);
                //释放人脸识别对象
                faceFactory?.Dispose();
                faceFactory = null;
                //token释放
                token.Dispose();
                token = null;
            }
            finally
            {
                FormHelper.SetButtonText(ButtonStart, _disableBtnString);
                FormHelper.SetControlStatus(this.ButtonStart, true);
                FormHelper.SetControlStatus(this.comboBox1, true);
                FormHelper.SetControlStatus(this.comboBox2, true);
                FormHelper.SetControlStatus(this.ButtonSave, false);
            }
        }

        /// <summary>
        /// 持续检测一次人脸，直到停止。
        /// </summary>
        /// <param name="token">取消标记</param>
        private async void StartDetector(CancellationToken token)
        {
            List<double> fpsList = new List<double>();
            double fps = 0;
            Stopwatch stopwatchFPS = new Stopwatch();
            Stopwatch stopwatch = new Stopwatch();
            isDetecting = true;
            try
            {
                while (VideoPlayer.IsRunning && !token.IsCancellationRequested)
                {
                    try
                    {
                        if (CheckBoxFPS.Checked)
                        {
                            stopwatch.Restart();
                            if (!stopwatchFPS.IsRunning)
                            { stopwatchFPS.Start(); }
                        }
                        Bitmap bitmap = VideoPlayer.GetCurrentVideoFrame(); // 获取摄像头画面 
                        if (bitmap == null)
                        {
                            await Task.Delay(10, token);
                            FormHelper.SetPictureBoxImage(FacePictureBox, bitmap);
                            continue;
                        }
                        if (!CheckBoxDetect.Checked)
                        {
                            await Task.Delay(1000 / 60, token);
                            FormHelper.SetPictureBoxImage(FacePictureBox, bitmap);
                            continue;
                        }
                        List<Models.VideoFaceInfo> faceInfos = new List<Models.VideoFaceInfo>();
                        using (FaceImage faceImage = bitmap.ToFaceImage())
                        {
                            var infos = await faceFactory.Get<FaceTracker>().TrackAsync(faceImage);
                            for (int i = 0; i < infos.Length; i++)
                            {

                                Models.VideoFaceInfo faceInfo = new Models.VideoFaceInfo
                                {
                                    Pid = infos[i].Pid,
                                    Location = infos[i].Location
                                };
                                if (CheckBoxFaceMask.Checked || CheckBoxFaceProperty.Checked)
                                {
                                    if (CheckBoxFaceMask.Checked)
                                    {
                                        var maskStatus = await faceFactory.Get<MaskDetector>().PlotMaskAsync(faceImage, infos[i]);
                                        faceInfo.HasMask = maskStatus.Masked;
                                    }
                                    if (CheckBoxFaceProperty.Checked)
                                    {
                                        FaceRecognizer faceRecognizer = null;
                                        if (faceInfo.HasMask)
                                        {
                                            faceRecognizer = faceFactory.GetFaceRecognizerWithMask();
                                        }
                                        else
                                        {
                                            faceRecognizer = faceFactory.Get<FaceRecognizer>();
                                        }
                                        var points = await faceFactory.Get<FaceLandmarker>().MarkAsync(faceImage, infos[i]);
                                        float[] extractData = await faceRecognizer.ExtractAsync(faceImage, points);

                                        UserInfo userInfo = CacheManager.Instance.Get(faceRecognizer, extractData);
                                        if (userInfo != null)
                                        {
                                            faceInfo.Name = userInfo.Name;
                                            faceInfo.Age = userInfo.Age;
                                            switch (userInfo.Gender)
                                            {
                                                case GenderEnum.Male:
                                                    faceInfo.Gender = Gender.Male;
                                                    break;
                                                case GenderEnum.Female:
                                                    faceInfo.Gender = Gender.Female;
                                                    break;
                                                case GenderEnum.Unknown:
                                                    faceInfo.Gender = Gender.Unknown;
                                                    break;
                                            }
                                        }
                                        else
                                        {
                                            faceInfo.Age = await faceFactory.Get<AgePredictor>().PredictAgeWithCropAsync(faceImage, points);
                                            faceInfo.Gender = await faceFactory.Get<GenderPredictor>().PredictGenderWithCropAsync(faceImage, points);
                                        }
                                    }
                                }
                                faceInfos.Add(faceInfo);
                            }
                        }
                        using (Graphics g = Graphics.FromImage(bitmap))
                        {
                            if (faceInfos.Any()) // 如果有人脸，在 bitmap 上绘制出人脸的位置信息
                            {
                                g.DrawRectangles(new Pen(Color.Red, 4), faceInfos.Select(p => p.Rectangle).ToArray());
                                if (CheckBoxDetect.Checked)
                                {
                                    for (int i = 0; i < faceInfos.Count; i++)
                                    {
                                        StringBuilder builder = new StringBuilder();
                                        if (CheckBoxFaceMask.Checked || CheckBoxFaceProperty.Checked)
                                        {
                                            builder.Append($"Pid: {faceInfos[i].Pid}");
                                            builder.Append(" | ");
                                        }
                                        if (CheckBoxFaceMask.Checked)
                                        {
                                            builder.Append($"口罩：{(faceInfos[i].HasMask ? "是" : "否")}");
                                            if (CheckBoxFaceProperty.Checked)
                                            {
                                                builder.Append(" | ");
                                            }
                                        }
                                        if (CheckBoxFaceProperty.Checked)
                                        {
                                            if (!string.IsNullOrEmpty(faceInfos[i].Name))
                                            {
                                                builder.Append(faceInfos[i].Name);
                                                builder.Append(" | ");
                                            }
                                            builder.Append($"{faceInfos[i].Age} 岁");
                                            builder.Append(" | ");
                                            builder.Append($"{faceInfos[i].GenderDescribe}");
                                            builder.Append(" | ");
                                        }
                                        g.DrawString(builder.ToString(), new Font("微软雅黑", 24), Brushes.Green, new PointF(faceInfos[i].Location.X + faceInfos[i].Location.Width + 24, faceInfos[i].Location.Y));
                                    }
                                }
                            }
                            if (CheckBoxFPS.Checked)
                            {
                                stopwatch.Stop();

                                if (numericUpDownFPSTime.Value > 0)
                                {
                                    fpsList.Add(1000f / stopwatch.ElapsedMilliseconds);
                                    if (stopwatchFPS.ElapsedMilliseconds >= numericUpDownFPSTime.Value)
                                    {
                                        fps = fpsList.Average();
                                        fpsList.Clear();
                                        stopwatchFPS.Reset();
                                    }
                                }
                                else
                                {
                                    fps = 1000f / stopwatch.ElapsedMilliseconds;
                                }
                                g.DrawString($"{fps:#.#} FPS", new Font("微软雅黑", 24), Brushes.Green, new Point(10, 10));
                            }
                        }
                        FormHelper.SetPictureBoxImage(FacePictureBox, bitmap);
                    }
                    catch (TaskCanceledException)
                    {
                        break;
                    }
                    catch { }
                }
            }
            finally
            {
                isDetecting = false;
            }
        }

        #endregion
    }
}
