using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ViewFaceCore.Demo.VideoForm.Utils
{
    public static class FormHelper
    {
        public static void SetControlStatus(Control control, bool status)
        {
            if (control.InvokeRequired)
            {
                control.Invoke(new Action(() =>
                {
                    control.Enabled = status;
                }));
            }
            else
            {
                control.Enabled = status;
            }
        }

        public static void SetLabelStatus(Label label, string message, bool status)
        {
            if (label.InvokeRequired)
            {
                label.Invoke(new Action(() =>
                {
                    label.Text = message;
                    label.Visible = true;
                }));
            }
            else
            {
                label.Text = message;
                label.Visible = true;
            }
        }

        public static void SetTextBoxText(TextBox tb, string text)
        {
            if (tb == null)
                return;
            if (tb.InvokeRequired)
            {
                tb.Invoke(new Action(() =>
                {
                    tb.Text = text;
                }));
            }
            else
            {
                tb.Text = text;
            }
        }

        public static void SetButtonText(Button tb, string text)
        {
            if (tb == null)
                return;
            if (tb.InvokeRequired)
            {
                tb.Invoke(new Action(() =>
                {
                    tb.Text = text;
                }));
            }
            else
            {
                tb.Text = text;
            }
        }

        public static void SetPictureBoxImage(PictureBox pictureBox, Bitmap bitmap)
        {
            if (pictureBox.InvokeRequired)
            {
                pictureBox.Invoke(new Action(() =>
                {
                    try
                    {
                        pictureBox.Image?.Dispose();
                    }
                    catch { }
                    pictureBox.Image = bitmap;
                }));
            }
            else
            {
                try
                {
                    pictureBox.Image?.Dispose();
                }
                catch { }
                pictureBox.Image = bitmap;
            }
        }
    }
}
