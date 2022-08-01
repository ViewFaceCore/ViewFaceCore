using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using ViewFaceCore.Core;

namespace ViewFaceCore.IISWebApp.Test.Controllers
{
    public class HomeController : Controller
    {
        private readonly static string imagePath = @"images\Jay_3.jpg";
        private readonly FaceDetector _faceDetector = null;

        public HomeController()
        {
            _faceDetector = new FaceDetector();
        }

        public ActionResult Index()
        {
            using (var bitmap = SKBitmap.Decode(imagePath))
            {
                var result = _faceDetector.Detect(bitmap);
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