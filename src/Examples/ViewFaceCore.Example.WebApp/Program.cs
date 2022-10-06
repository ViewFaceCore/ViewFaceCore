using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using ViewFaceCore.Configs;
using ViewFaceCore.Extension.DependencyInjection;

namespace ViewFaceCore.Example.WebApp
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            //添加人脸识别能力
            builder.Services.AddViewFaceCore(p =>
            {
                p.FaceTrackerConfig = new FaceTrackerConfig(1280, 720);
                //启用所有能力（不建议，会占用大量内存，按需启用即可）
                p.IsEnableAll = true;
            });

            // Add services to the container.
            builder.Services.AddControllersWithViews()
                .AddRazorRuntimeCompilation();

            var app = builder.Build();

            // Configure the HTTP request pipeline.
            if (!app.Environment.IsDevelopment())
            {
                app.UseExceptionHandler("/Home/Error");
                // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
                app.UseHsts();
            }

            app.UseHttpsRedirection();
            app.UseStaticFiles();

            app.UseRouting();

            app.UseAuthorization();

            app.MapControllerRoute(
                name: "default",
                pattern: "{controller=Home}/{action=Index}/{id?}");

            app.Run();
        }
    }
}