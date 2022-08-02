using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using ViewFaceCore.Core;
using ViewFaceCore.IISWebApp.Test.Services;

namespace ViewFaceCore.IISWebApp.Test.Controllers
{
    public class HomeController : Controller
    {
        private readonly static string _imagePath = @"images\Jay_3.jpg";

        public ActionResult Index()
        {
            string imagePath = Path.Combine(Server.MapPath("/"), "Content", _imagePath);
            if (!System.IO.File.Exists(imagePath))
            {
                return Content($"图片'_imagePath'不存在！");
            }
            using (var bitmap = (Bitmap)Image.FromFile(imagePath))
            {
                var result = ViewFaceCoreService.Instance.FaceDetector.Detect(bitmap);
                ViewBag.Message = $"当前进程名称：{Process.GetCurrentProcess().ProcessName}，识别到了{result.Length}个人脸信息，人脸信息：{(result.Length > 0 ? result[0].ToString() : "莫球得")}";
                return View();
            }
        }

        public ActionResult About()
        {
            ViewBag.Message = "Your application description page.";

            return View();
        }

        public ActionResult Contact()
        {
            ViewBag.Message = "Your contact page.";

            return View();
        }
    }
}