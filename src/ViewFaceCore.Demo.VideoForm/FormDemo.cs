using Accord.Video.DirectShow;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using ViewFaceCore.Configs;
using ViewFaceCore.Core;
using ViewFaceCore.Extensions;
using ViewFaceCore.Model;

namespace ViewFaceCore.Demo.VideoForm
{
    public partial class FormDemo : Form
    {
        public FormDemo()
        {
            InitializeComponent();
            // 隐藏摄像头画面控件
            VideoPlayer.Visible = false;
        }

        /// <summary>
        /// 摄像头设备信息集合
        /// </summary>
        FilterInfoCollection VideoDevices;

        /// <summary>
        /// 取消令牌
        /// </summary>
        CancellationTokenSource Token { get; set; }

        ViewFaceFactory faceFactory = new ViewFaceFactory(1280, 720);

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
            VideoDevices = new FilterInfoCollection(FilterCategory.VideoInputDevice);
            comboBox1.Items.Clear();
            foreach (FilterInfo info in VideoDevices)
            {
                comboBox1.Items.Add(info.Name);
            }
            if (comboBox1.Items.Count > 0)
            { comboBox1.SelectedIndex = 0; }
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
            }
            else
            {
                if (comboBox1.SelectedIndex == -1) return;
                FilterInfo info = VideoDevices[comboBox1.SelectedIndex];
                VideoCaptureDevice videoCapture = new VideoCaptureDevice(info.MonikerString);
                VideoPlayer.VideoSource = videoCapture;
                VideoPlayer.Start();
                ButtonStart.Text = "关闭摄像头";
                Token = new CancellationTokenSource();
                StartDetector(Token.Token);
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
                List<FaceInfo> faceInfos = new List<FaceInfo>();
                var infos = await faceFactory.Get<FaceTracker>().TrackAsync(bitmap);
                for (int i = 0; i < infos.Length; i++)
                {
                    FaceInfo faceInfo = new FaceInfo
                    {
                        Pid = infos[i].Pid,
                        Location = infos[i].Location
                    };
                    if (CheckBoxFaceMask.Checked || CheckBoxFaceProperty.Checked)
                    {
                        Model.FaceInfo info = infos[i].ToFaceInfo();
                        if (CheckBoxFaceMask.Checked)
                        {
                            var maskStatus = faceFactory.Get<MaskDetector>().PlotMask(bitmap, info);
                            faceInfo.HasMask = maskStatus != null && maskStatus.Status ? maskStatus.Score > 0.5 : false;
                        }
                        if (CheckBoxFaceProperty.Checked)
                        {
                            var points = await faceFactory.Get<FaceLandmarker>().MarkAsync(bitmap, info);
                            faceInfo.Age = await faceFactory.Get<AgePredictor>().PredictAgeAsync(bitmap, points);
                            faceInfo.Gender = await faceFactory.Get<GenderPredictor>().PredictGenderAsync(bitmap, points);
                        }
                    }
                    faceInfos.Add(faceInfo);
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
    }
}
