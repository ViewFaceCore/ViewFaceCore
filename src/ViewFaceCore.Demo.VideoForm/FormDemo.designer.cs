namespace ViewFaceCore.Demo.VideoForm
{
    partial class FormDemo
    {
        /// <summary>
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清理所有正在使用的资源。
        /// </summary>
        /// <param name="disposing">如果应释放托管资源，为 true；否则为 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows 窗体设计器生成的代码

        /// <summary>
        /// 设计器支持所需的方法 - 不要修改
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.ButtonStart = new System.Windows.Forms.Button();
            this.comboBox1 = new System.Windows.Forms.ComboBox();
            this.FacePictureBox = new System.Windows.Forms.PictureBox();
            this.VideoPlayer = new Accord.Controls.VideoSourcePlayer();
            this.CheckBoxFaceProperty = new System.Windows.Forms.CheckBox();
            this.toolTip1 = new System.Windows.Forms.ToolTip(this.components);
            this.CheckBoxFPS = new System.Windows.Forms.CheckBox();
            this.CheckBoxDetect = new System.Windows.Forms.CheckBox();
            this.numericUpDownFPSTime = new System.Windows.Forms.NumericUpDown();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.CheckBoxFaceMask = new System.Windows.Forms.CheckBox();
            ((System.ComponentModel.ISupportInitialize)(this.FacePictureBox)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownFPSTime)).BeginInit();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.SuspendLayout();
            // 
            // ButtonStart
            // 
            this.ButtonStart.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.ButtonStart.Location = new System.Drawing.Point(728, 10);
            this.ButtonStart.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.ButtonStart.Name = "ButtonStart";
            this.ButtonStart.Size = new System.Drawing.Size(109, 49);
            this.ButtonStart.TabIndex = 1;
            this.ButtonStart.Text = "打开摄像头并识别人脸";
            this.ButtonStart.UseVisualStyleBackColor = true;
            this.ButtonStart.Click += new System.EventHandler(this.ButtonStart_Click);
            // 
            // comboBox1
            // 
            this.comboBox1.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.comboBox1.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBox1.FormattingEnabled = true;
            this.comboBox1.Location = new System.Drawing.Point(728, 63);
            this.comboBox1.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.comboBox1.Name = "comboBox1";
            this.comboBox1.Size = new System.Drawing.Size(109, 20);
            this.comboBox1.TabIndex = 2;
            // 
            // FacePictureBox
            // 
            this.FacePictureBox.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.FacePictureBox.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.FacePictureBox.Location = new System.Drawing.Point(9, 10);
            this.FacePictureBox.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.FacePictureBox.Name = "FacePictureBox";
            this.FacePictureBox.Size = new System.Drawing.Size(703, 401);
            this.FacePictureBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.FacePictureBox.TabIndex = 4;
            this.FacePictureBox.TabStop = false;
            // 
            // VideoPlayer
            // 
            this.VideoPlayer.Location = new System.Drawing.Point(20, 18);
            this.VideoPlayer.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.VideoPlayer.Name = "VideoPlayer";
            this.VideoPlayer.Size = new System.Drawing.Size(181, 134);
            this.VideoPlayer.TabIndex = 5;
            this.VideoPlayer.Text = "videoSourcePlayer1";
            this.VideoPlayer.VideoSource = null;
            // 
            // CheckBoxFaceProperty
            // 
            this.CheckBoxFaceProperty.AutoSize = true;
            this.CheckBoxFaceProperty.Location = new System.Drawing.Point(5, 37);
            this.CheckBoxFaceProperty.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.CheckBoxFaceProperty.Name = "CheckBoxFaceProperty";
            this.CheckBoxFaceProperty.Size = new System.Drawing.Size(96, 16);
            this.CheckBoxFaceProperty.TabIndex = 6;
            this.CheckBoxFaceProperty.Text = "检测人脸属性";
            this.toolTip1.SetToolTip(this.CheckBoxFaceProperty, "启用检测人脸属性后，图片的刷新率可能会显著下降");
            this.CheckBoxFaceProperty.UseVisualStyleBackColor = true;
            // 
            // CheckBoxFPS
            // 
            this.CheckBoxFPS.AutoSize = true;
            this.CheckBoxFPS.Checked = true;
            this.CheckBoxFPS.CheckState = System.Windows.Forms.CheckState.Checked;
            this.CheckBoxFPS.Location = new System.Drawing.Point(5, 77);
            this.CheckBoxFPS.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.CheckBoxFPS.Name = "CheckBoxFPS";
            this.CheckBoxFPS.Size = new System.Drawing.Size(72, 16);
            this.CheckBoxFPS.TabIndex = 7;
            this.CheckBoxFPS.Text = "计算 FPS";
            this.CheckBoxFPS.UseVisualStyleBackColor = true;
            // 
            // CheckBoxDetect
            // 
            this.CheckBoxDetect.AutoSize = true;
            this.CheckBoxDetect.Checked = true;
            this.CheckBoxDetect.CheckState = System.Windows.Forms.CheckState.Checked;
            this.CheckBoxDetect.Location = new System.Drawing.Point(5, 17);
            this.CheckBoxDetect.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.CheckBoxDetect.Name = "CheckBoxDetect";
            this.CheckBoxDetect.Size = new System.Drawing.Size(72, 16);
            this.CheckBoxDetect.TabIndex = 8;
            this.CheckBoxDetect.Text = "人脸检测";
            this.CheckBoxDetect.UseVisualStyleBackColor = true;
            this.CheckBoxDetect.CheckedChanged += new System.EventHandler(this.CheckBoxDetect_CheckedChanged);
            // 
            // numericUpDownFPSTime
            // 
            this.numericUpDownFPSTime.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.numericUpDownFPSTime.Increment = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numericUpDownFPSTime.Location = new System.Drawing.Point(4, 31);
            this.numericUpDownFPSTime.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.numericUpDownFPSTime.Maximum = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            this.numericUpDownFPSTime.Name = "numericUpDownFPSTime";
            this.numericUpDownFPSTime.Size = new System.Drawing.Size(92, 21);
            this.numericUpDownFPSTime.TabIndex = 9;
            this.numericUpDownFPSTime.Value = new decimal(new int[] {
            200,
            0,
            0,
            0});
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.numericUpDownFPSTime);
            this.groupBox1.Location = new System.Drawing.Point(4, 98);
            this.groupBox1.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Padding = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.groupBox1.Size = new System.Drawing.Size(101, 56);
            this.groupBox1.TabIndex = 11;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "FPS 刷新时间 (单位:毫秒)";
            // 
            // groupBox2
            // 
            this.groupBox2.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.groupBox2.Controls.Add(this.CheckBoxFaceMask);
            this.groupBox2.Controls.Add(this.CheckBoxFaceProperty);
            this.groupBox2.Controls.Add(this.groupBox1);
            this.groupBox2.Controls.Add(this.CheckBoxFPS);
            this.groupBox2.Controls.Add(this.CheckBoxDetect);
            this.groupBox2.Location = new System.Drawing.Point(728, 88);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(109, 160);
            this.groupBox2.TabIndex = 12;
            this.groupBox2.TabStop = false;
            // 
            // CheckBoxFaceMask
            // 
            this.CheckBoxFaceMask.AutoSize = true;
            this.CheckBoxFaceMask.Checked = true;
            this.CheckBoxFaceMask.CheckState = System.Windows.Forms.CheckState.Checked;
            this.CheckBoxFaceMask.Location = new System.Drawing.Point(5, 57);
            this.CheckBoxFaceMask.Margin = new System.Windows.Forms.Padding(2);
            this.CheckBoxFaceMask.Name = "CheckBoxFaceMask";
            this.CheckBoxFaceMask.Size = new System.Drawing.Size(84, 16);
            this.CheckBoxFaceMask.TabIndex = 12;
            this.CheckBoxFaceMask.Text = "戴口罩检测";
            this.toolTip1.SetToolTip(this.CheckBoxFaceMask, "启用检测人脸属性后，图片的刷新率可能会显著下降");
            this.CheckBoxFaceMask.UseVisualStyleBackColor = true;
            // 
            // FormDemo
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(845, 421);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.VideoPlayer);
            this.Controls.Add(this.comboBox1);
            this.Controls.Add(this.ButtonStart);
            this.Controls.Add(this.FacePictureBox);
            this.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.Name = "FormDemo";
            this.Text = "Form1";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Form_Closing);
            this.Load += new System.EventHandler(this.Form_Load);
            ((System.ComponentModel.ISupportInitialize)(this.FacePictureBox)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownFPSTime)).EndInit();
            this.groupBox1.ResumeLayout(false);
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion
        private System.Windows.Forms.Button ButtonStart;
        private System.Windows.Forms.ComboBox comboBox1;
        private System.Windows.Forms.PictureBox FacePictureBox;
        private Accord.Controls.VideoSourcePlayer VideoPlayer;
        private System.Windows.Forms.CheckBox CheckBoxFaceProperty;
        private System.Windows.Forms.ToolTip toolTip1;
        private System.Windows.Forms.CheckBox CheckBoxFPS;
        private System.Windows.Forms.CheckBox CheckBoxDetect;
        private System.Windows.Forms.NumericUpDown numericUpDownFPSTime;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.CheckBox CheckBoxFaceMask;
    }
}

