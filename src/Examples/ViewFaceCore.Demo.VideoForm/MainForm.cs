using AForge.Video.DirectShow;
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
        private static int _videoWidth = 1280;
        private static int _videoHeight = 720;

        public MainForm()
        {
            InitializeComponent();

        }

        /// <summary>
        /// 摄像头设备信息集合
        /// </summary>
        FilterInfoCollection VideoDevices;

        /// <summary>
        /// 取消令牌
        /// </summary>
        CancellationTokenSource Token { get; set; }

        ViewFaceFactory faceFactory = new ViewFaceFactory(_videoWidth, _videoHeight);

        /// <summary>
        /// 指示是否应关闭窗体
        /// </summary>
        bool IsClose = false;

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
            VideoDevices = new FilterInfoCollection(FilterCategory.VideoInputDevice);
            comboBox1.Items.Clear();
            foreach (FilterInfo info in VideoDevices)
            {
                comboBox1.Items.Add(info.Name);
            }
            if (comboBox1.Items.Count > 0)
            {
                comboBox1.SelectedIndex = 0;
            }
            //默认禁用拍照按钮
            ButtonSave.Enabled = false;

            ButtonStart_Click(null, null);
        }

        /// <summary>
        /// 窗体关闭时，关闭摄像头
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Form_Closing(object sender, FormClosingEventArgs e)
        {
            //释放人脸识别对象
            faceFactory.Dispose();
            Token?.Cancel();
            if (!IsClose && VideoPlayer.IsRunning)
            { // 若摄像头开启时，点击关闭是暂不关闭，并设置关闭窗口的标识，待摄像头等设备关闭后，再关闭窗体。
                e.Cancel = true;
                IsClose = true;
            }
        }

        /// <summary>
        /// 点击开始按钮时
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void ButtonStart_Click(object sender, EventArgs e)
        {
            if (VideoPlayer.IsRunning)
            {
                Token?.Cancel();
                ButtonStart.Text = "打开摄像头并识别人脸";
                ButtonSave.Enabled = false;
            }
            else
            {
                if (comboBox1.SelectedIndex == -1)
                    return;
                FilterInfo info = VideoDevices[comboBox1.SelectedIndex];
                VideoCaptureDevice videoCapture = new VideoCaptureDevice(info.MonikerString);
                var videoResolution = videoCapture.VideoCapabilities.Where(p => p.FrameSize.Width == _videoWidth && p.FrameSize.Height == _videoHeight).FirstOrDefault();
                if (videoResolution == null)
                {
                    MessageBox.Show($"摄像头不支持拍摄分辨率为{_videoHeight}x{_videoHeight}的视频，请重新指定分辨率，或更换摄像头！", "警告", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
                videoCapture.VideoResolution = videoResolution;
                VideoPlayer.VideoSource = videoCapture;
                VideoPlayer.Start();

                ButtonStart.Text = "关闭摄像头";
                Token = new CancellationTokenSource();

                StartDetector(Token.Token);
                ButtonSave.Enabled = true;
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
            while (VideoPlayer.IsRunning && !token.IsCancellationRequested)
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
                    await Task.Delay(10);
                    PictureBoxDispose(bitmap);
                    continue;
                }
                if (!CheckBoxDetect.Checked)
                {
                    await Task.Delay(1000 / 60);
                    PictureBoxDispose(bitmap);
                    continue;
                }
                List<Models.FaceInfo> faceInfos = new List<Models.FaceInfo>();
                using (FaceImage faceImage = bitmap.ToFaceImage())
                {
                    var infos = await Task.Run(() => faceFactory.Get<FaceTracker>().Track(faceImage));
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
                                var maskStatus = await Task.Run(() => faceFactory.Get<MaskDetector>().PlotMask(faceImage, info));
                                faceInfo.HasMask = maskStatus.Masked;
                            }
                            if (CheckBoxFaceProperty.Checked)
                            {
                                var points = await Task.Run(() => faceFactory.Get<FaceLandmarker>().Mark(faceImage, info));
                                faceInfo.Age = await Task.Run(() => faceFactory.Get<AgePredictor>().PredictAge(faceImage, points));
                                faceInfo.Gender = await Task.Run(() => faceFactory.Get<GenderPredictor>().PredictGender(faceImage, points));
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
                PictureBoxDispose(bitmap);
            }

            VideoPlayer?.SignalToStop();
            VideoPlayer?.WaitForStop();
            FacePictureBox.Image = null;
            if (IsClose)
            {
                Close();
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
        }

        private void PictureBoxDispose(Bitmap bitmap)
        {
            FacePictureBox.Image?.Dispose();
            FacePictureBox.Image = bitmap;
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

        private bool _isTakingPicture = false;
        private readonly static object _locker = new object();

        private ConcurrentQueue<TakePhotoInfo> takePhotos = new ConcurrentQueue<TakePhotoInfo>();

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
                    MessageBox.Show("拍照失败，请重试！", "警告", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }
                if (!takePhotos.TryDequeue(out TakePhotoInfo takePhotoInfo))
                {
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
                Action setBtnEnable = new Action(() => { ButtonSave.Enabled = true; });
                if (ButtonSave.InvokeRequired)
                    ButtonSave.Invoke(setBtnEnable);
                else
                    setBtnEnable();

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

        private void SetTakingPictureStatus(bool status)
        {
            lock (_locker)
            {
                _isTakingPicture = status;
            }
        }

        public Bitmap DeepClone(Bitmap bitmap)
        {
            Bitmap dstBitmap = null;
            using (MemoryStream mStream = new MemoryStream())
            {
                BinaryFormatter bf = new BinaryFormatter();
                bf.Serialize(mStream, bitmap);
                mStream.Seek(0, SeekOrigin.Begin);//指定当前流的位置为流的开头。
                dstBitmap = (Bitmap)bf.Deserialize(mStream);
                mStream.Close();
            }
            return dstBitmap;

        }
    }
}
