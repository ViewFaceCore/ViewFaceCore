using AForge.Video.DirectShow;
using Microsoft.Extensions.Configuration;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using ViewFaceCore.Configs;
using ViewFaceCore.Core;
using ViewFaceCore.Demo.VideoForm.Extensions;
using ViewFaceCore.Demo.VideoForm.Models;
using ViewFaceCore.Extensions;
using ViewFaceCore.Model;

namespace ViewFaceCore.Demo.VideoForm
{
    public partial class MainForm : Form
    {
        private const string _enableBtnText = "关闭摄像头";
        private const string _disableBtnText = "打开摄像头并识别人脸";

        /// <summary>
        /// 摄像头设备信息集合
        /// </summary>
        private FilterInfoCollection _videoDevices;

        private ViewFaceFactory _faceFactory;

        private CameraSettings _cameraSettings;

        private VideoCaptureDevice videoCapture = null;
        List<double> fpsList = new List<double>();
        double fps = 0;
        Stopwatch stopwatchFPS = new Stopwatch();
        Stopwatch stopwatch = new Stopwatch();

        public MainForm()
        {
            InitializeComponent();

            //加载配置
            IConfigurationRoot configuration = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json")
                .Build();
            _cameraSettings = configuration.GetSection("CameraSettings").Get<CameraSettings>();
            if (_cameraSettings == null || _cameraSettings.Height == 0 || _cameraSettings.Width == 0)
            {
                throw new Exception("加载配置文件appsettings.json失败，请检查配置文件是否存在！");
            }
        }

        #region Events

        /// <summary>
        /// 窗体加载时
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Form_Load(object sender, EventArgs e)
        {
            //初始化VideoDevices
            _videoDevices = new FilterInfoCollection(FilterCategory.VideoInputDevice);

            comboBox1.Items.Clear();
            foreach (FilterInfo info in _videoDevices)
            {
                comboBox1.Items.Add(info.Name);
            }
            if (comboBox1.Items.Count > 0)
            {
                comboBox1.SelectedIndex = 0;
            }
            ButtonStart.Text = _disableBtnText;
            //默认禁用拍照按钮
            ButtonSave.Enabled = false;

            if (_videoDevices.Count > 0)
            {
                _faceFactory = new ViewFaceFactory(_cameraSettings.Width, _cameraSettings.Height);
            }
            CheckBoxDetect.Checked = false;
            //ButtonStart_Click(null, null);
        }

        /// <summary>
        /// 窗体关闭时，关闭摄像头
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Form_Closing(object sender, FormClosingEventArgs e)
        {
            Thread.Sleep(100);
            //释放人脸识别对象
            _faceFactory?.Dispose();
        }

        /// <summary>
        /// 点击开始按钮时
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void ButtonStart_Click(object sender, EventArgs e)
        {
            if (ButtonStart.Text == _enableBtnText)
            {
                Stop();
            }
            else if (ButtonStart.Text == _disableBtnText)
            {
                Start();
            }
            else
            {
                MessageBox.Show($"Emmmm...姿势不对~~~", "警告", MessageBoxButtons.OK, MessageBoxIcon.Error);
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
            if (_isTakingPicture)
            {
                MessageBox.Show("拍照中...请稍后再试！", "警告", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            ButtonSave.Enabled = false;
            _ = Task.Run(() =>
            {
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();
                SetTakingPictureStatus(true);
                while (stopwatch.ElapsedMilliseconds < 3000 && _isTakingPicture)
                {
                    Thread.Sleep(1);
                }
                if (takePhotos.IsEmpty)
                {
                    SetButtonStatus(ButtonSave, true);
                    SetTakingPictureStatus(false);
                    MessageBox.Show("拍照失败，请重试！", "警告", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }
                if (!takePhotos.TryDequeue(out TakePhotoInfo takePhotoInfo))
                {
                    SetButtonStatus(ButtonSave, true);
                    SetTakingPictureStatus(false);
                    MessageBox.Show("拍照失败，请重试！", "警告", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }
                if (takePhotos.Count > 0)
                {
                    int count = takePhotos.Count;
                    for (int i = 0; i < count; i++)
                    {
                        takePhotos.TryDequeue(out _);
                    }
                }
                SetButtonStatus(ButtonSave, true);
                if (takePhotoInfo.FaceTrackInfos == null || !takePhotoInfo.FaceTrackInfos.Any())
                {
                    _ = MessageBox.Show("未识别到任何人脸信息！", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }
                //打开保存框
                UserInfoForm saveUser = new UserInfoForm(takePhotoInfo);
                saveUser.ShowDialog();
            });
        }


        #endregion

        private void Stop()
        {
            if (videoCapture == null)
            {
                SetButtonText(this.ButtonStart, _disableBtnText);
                SetButtonStatus(this.ButtonSave, false);
            }
            //表示停止
            if (videoCapture.IsRunning)
            {
                videoCapture.SignalToStop();
                videoCapture.WaitForStop();
                //置空
                DrawPictureBox(null);
            }
            videoCapture = null;
            SetButtonText(this.ButtonStart, _disableBtnText);
            SetButtonStatus(this.ButtonSave, false);
            SetButtonStatus(this.ButtonStart, true);
        }

        private void Start()
        {
            if (videoCapture != null)
            {
                Stop();
            }
            if (comboBox1.SelectedIndex == -1)
            {
                MessageBox.Show($"没有找到可用的摄像头！", "警告", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            videoCapture = new VideoCaptureDevice(_videoDevices[comboBox1.SelectedIndex].MonikerString);
            var videoResolution = videoCapture.VideoCapabilities.Where(p => p.FrameSize.Width == _cameraSettings.Width && p.FrameSize.Height == _cameraSettings.Height).FirstOrDefault();
            if (videoResolution == null)
            {
                List<string> supports = videoCapture.VideoCapabilities.OrderBy(p => p.FrameSize.Width).Select(p => $"{p.FrameSize.Width}x{p.FrameSize.Height}").ToList();
                string supportStr = "无，或获取失败";
                if (supports.Any())
                {
                    supportStr = string.Join("|", supports);
                }
                MessageBox.Show($"摄像头不支持拍摄分辨率为{_cameraSettings.Width}x{_cameraSettings.Height}的视频，请重新指定分辨率。\n支持分辨率：{supportStr}", "警告", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            videoCapture.VideoResolution = videoResolution;
            videoCapture.NewFrame += CaptureFrameEvent;

            //等有视频帧之后在启用按钮
            SetButtonStatus(this.ButtonStart, false);
            SetButtonStatus(this.ButtonSave, false);
            SetButtonText(this.ButtonStart, _enableBtnText);

            videoCapture.Start();
        }

        private bool _isTakingPicture = false;
        private readonly static object _locker = new object();

        private ConcurrentQueue<TakePhotoInfo> takePhotos = new ConcurrentQueue<TakePhotoInfo>();
        
        private async void CaptureFrameEvent(object sender, AForge.Video.NewFrameEventArgs eventArgs)
        {
            if (this.ButtonStart.Enabled == false)
            {
                SetButtonStatus(this.ButtonStart, true);
            }
            if (this.ButtonSave.Enabled == false)
            {
                SetButtonStatus(this.ButtonSave, true);
            }
            if (CheckBoxFPS.Checked)
            {
                stopwatch.Restart();
                if (!stopwatchFPS.IsRunning)
                {
                    stopwatchFPS.Start();
                }
            }
            using (Bitmap bitmap = eventArgs.Frame.DeepClone())
            {
                if (bitmap == null)
                {
                    DrawPictureBox(bitmap);
                    return;
                }
                if (!CheckBoxDetect.Checked)
                {
                    DrawPictureBox(bitmap);
                    return;
                }
                List<Models.FaceInfo> faceInfos = new List<Models.FaceInfo>();
                using (FaceImage faceImage = bitmap.ToFaceImage())
                {
                    var infos = await Task.Run(() => _faceFactory.Get<FaceTracker>().Track(faceImage));
                    //拍摄照片
                    if (_isTakingPicture)
                    {
                        takePhotos.Enqueue(new TakePhotoInfo()
                        {
                            FaceTrackInfos = infos,
                            Bitmap = bitmap.DeepClone(),
                        });
                        SetTakingPictureStatus(false);
                    }
                    for (int i = 0; i < infos.Length; i++)
                    {
                        Models.FaceInfo faceInfo = new Models.FaceInfo
                        {
                            Pid = infos[i].Pid,
                            Location = infos[i].Location
                        };
                        if (CheckBoxFaceMask.Checked || CheckBoxFaceProperty.Checked)
                        {
                            Model.FaceInfo info = infos[i].ToFaceInfo();
                            if (CheckBoxFaceMask.Checked)
                            {
                                var maskStatus = await Task.Run(() => _faceFactory.Get<MaskDetector>().PlotMask(faceImage, info));
                                faceInfo.HasMask = maskStatus.Masked;
                            }
                            if (CheckBoxFaceProperty.Checked)
                            {
                                var points = await Task.Run(() => _faceFactory.Get<FaceLandmarker>().Mark(faceImage, info));
                                faceInfo.Age = await Task.Run(() => _faceFactory.Get<AgePredictor>().PredictAge(faceImage, points));
                                faceInfo.Gender = await Task.Run(() => _faceFactory.Get<GenderPredictor>().PredictGender(faceImage, points));
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
                                builder.Append($"Pid: {faceInfos[i].Pid}");
                                if (CheckBoxFaceMask.Checked || CheckBoxFaceProperty.Checked)
                                {
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
                                    builder.Append($"{faceInfos[i].Age} 岁");
                                    builder.Append(" | ");
                                    builder.Append($"{faceInfos[i].GenderDescribe}");
                                    builder.Append(" | ");
                                }
                                g.DrawString(builder.ToString(), new Font("微软雅黑", 24), Brushes.Green, new PointF(faceInfos[i].Location.X + faceInfos[i].Location.Width + 24, faceInfos[i].Location.Y));
                            }
                        }
                    }
                    //计算fps
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

                DrawPictureBox(bitmap);
            }
        }

        private void DrawPictureBox(Bitmap source)
        {
            if (FacePictureBox.InvokeRequired)
            {
                FacePictureBox.Invoke(new Action(() =>
                {
                    FacePictureBox.Image?.Dispose();
                    FacePictureBox.Image = source;
                    FacePictureBox.Refresh();
                }));
            }
            else
            {
                FacePictureBox.Image?.Dispose();
                FacePictureBox.Image = source;
                FacePictureBox.Refresh();
            }
        }

        private void SetTakingPictureStatus(bool status)
        {
            lock (_locker)
            {
                if (!status && !_isTakingPicture)
                {
                    return;
                }
                _isTakingPicture = status;
            }
        }

        private void SetButtonStatus(Button button, bool status)
        {
            Action setBtnEnable = new Action(() => { button.Enabled = status; });
            if (button.InvokeRequired)
                button.Invoke(setBtnEnable);
            else
                setBtnEnable();
        }

        private void SetButtonText(Button button, string text)
        {
            Action setBtnText = new Action(() => { button.Text = text; });
            if (button.InvokeRequired)
                button.Invoke(setBtnText);
            else
                button.Text = text;
        }
    }
}
