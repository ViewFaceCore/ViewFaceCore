using System.Web;
using System.Web.Mvc;

namespace ViewFaceCore.IISWebApp.Test
{
    public class FilterConfig
    {
        public static void RegisterGlobalFilters(GlobalFilterCollection filters)
        {
            filters.Add(new HandleErrorAttribute());
        }
    }
}
