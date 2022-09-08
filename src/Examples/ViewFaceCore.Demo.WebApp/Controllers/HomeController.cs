using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using System.Diagnostics;
using ViewFaceCore.Core;
using ViewFaceCore.Example.WebApp.Models;
using ViewFaceCore.Extension.DependencyInjection;

namespace ViewFaceCore.Example.WebApp.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;

        public HomeController(ILogger<HomeController> logger, IViewFaceFactory faceFactory)
        {
            _logger = logger;

            FaceDetector faceDetector = faceFactory.Get<FaceDetector>();
        }

        public IActionResult Index()
        {
            return View();
        }

        public IActionResult Privacy()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}