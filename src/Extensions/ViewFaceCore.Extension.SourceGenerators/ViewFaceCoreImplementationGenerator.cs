using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace ViewFaceCore.Extension.SourceGenerators
{
    [Generator]
    public class ViewFaceCoreImplementationGenerator : ISourceGenerator
    {
        private TestSyntaxReceiver testSyntaxReceiver = new TestSyntaxReceiver();
        public void Initialize(GeneratorInitializationContext context)
        {
            context.RegisterForSyntaxNotifications(new SyntaxReceiverCreator(() => testSyntaxReceiver));

            context.RegisterForPostInitialization(_context =>
            {

            });
        }

        public void Execute(GeneratorExecutionContext context)
        {
            if (!testSyntaxReceiver.CanGenerate) { return; }
            string Image = testSyntaxReceiver.ImageTypeName;
            string namespaces = string.Join("\r\n", testSyntaxReceiver.NamespaceNames);

            string codeTemplate = $$"""
{{namespaces}}

namespace ViewFaceCore.Core
{

    /// <summary>
    /// ViewFaceCore 的 <see cref="{{Image}}"/> 实现
    /// </summary>
    public static class Extensions
    {
        /// <summary>
        /// 识别 <paramref name="image"/> 中的人脸，并返回人脸的信息。
        /// </summary>
        /// <param name="viewFace"></param>
        /// <param name="image">人脸图像信息</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="FaceDetector.DetectConfig"/> 重新检测。</returns>
        public static FaceInfo[] Detect(this FaceDetector viewFace, {{Image}} image)
        {
            using var faceImage = image.ToFaceImage();
            return viewFace.Detect(faceImage);
        }

        /// <summary>
        /// 识别 <paramref name="image"/> 中指定的人脸信息 <paramref name="info"/> 的关键点坐标。
        /// </summary>
        /// <param name="viewFace"></param>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">指定的人脸信息</param>
        /// <returns>若失败，则返回结果 Length == 0</returns>
        public static FaceMarkPoint[] Mark(this FaceLandmarker viewFace, {{Image}} image, FaceInfo info)
        {
            using var faceImage = image.ToFaceImage();
            return viewFace.Mark(faceImage, info);
        }

        /// <summary>
        /// 提取人脸特征值。
        /// </summary>
        /// <param name="viewFace"></param>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">人脸关键点数据</param>
        /// <returns></returns>
        public static float[] Extract(this FaceRecognizer viewFace, {{Image}} image, FaceMarkPoint[] points)
        {
            using var faceImage = image.ToFaceImage();
            return viewFace.Extract(faceImage, points);
        }

        /// <summary>
        /// 活体检测器。(单帧图片)
        /// </summary>
        /// <param name="viewFace"></param>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">面部信息<para>通过 <see cref="Detect(FaceDetector, {{Image}})"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="Mark(FaceLandmarker, {{Image}}, FaceInfo)"/> 获取</para></param>
        /// <returns>活体检测状态</returns>
        public static AntiSpoofingResult Predict(this FaceAntiSpoofing viewFace, {{Image}} image, FaceInfo info, FaceMarkPoint[] points)
        {
            using var faceImage = image.ToFaceImage();
            return viewFace.Predict(faceImage, info, points);
        }

        /// <summary>
        /// 活体检测器。(视频帧图片)
        /// </summary>
        /// <param name="viewFace"></param>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">面部信息<para>通过 <see cref="Detect(FaceDetector, {{Image}})"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="Mark(FaceLandmarker, {{Image}}, FaceInfo)"/> 获取</para></param>
        /// <returns>如果为 <see cref="AntiSpoofingStatus.Detecting"/>，则说明需要继续调用此方法，传入更多的图片</returns>
        public static AntiSpoofingResult PredictVideo(this FaceAntiSpoofing viewFace, {{Image}} image, FaceInfo info, FaceMarkPoint[] points)
        {
            using var faceImage = image.ToFaceImage();
            return viewFace.PredictVideo(faceImage, info, points);
        }

        /// <summary>
        /// 人脸质量评估
        /// </summary>
        /// <param name="viewFace"></param>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">面部信息<para>通过 <see cref="Detect(FaceDetector, {{Image}})"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="Mark(FaceLandmarker, {{Image}}, FaceInfo)"/> 获取</para></param>
        /// <param name="type">质量评估类型</param>
        /// <returns></returns>
        public static QualityResult Detect(this FaceQuality viewFace, {{Image}} image, FaceInfo info, FaceMarkPoint[] points, QualityType type)
        {
            using var faceImage = image.ToFaceImage();
            return viewFace.Detect(faceImage, info, points, type);
        }

        /// <summary>
        /// 年龄预测（自动裁剪）
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.age_predictor">age_predictor.csta</a>
        /// </para>
        /// </summary>
        /// <param name="viewFace"></param>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">关键点坐标<para>通过 <see cref="Mark(FaceLandmarker, {{Image}}, FaceInfo)"/> 获取</para></param>
        /// <returns>-1: 预测失败失败，其它: 预测的年龄。</returns>
        public static int PredictAgeWithCrop(this AgePredictor viewFace, {{Image}} image, FaceMarkPoint[] points)
        {
            using var faceImage = image.ToFaceImage();
            return viewFace.PredictAgeWithCrop(faceImage, points);
        }

        /// <summary>
        /// 年龄预测
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.age_predictor">age_predictor.csta</a>
        /// </para>
        /// </summary>
        /// <param name="viewFace"></param>
        /// <param name="image">人脸图像信息</param>
        /// <returns>-1: 预测失败失败，其它: 预测的年龄。</returns>
        public static int PredictAge(this AgePredictor viewFace, {{Image}} image)
        {
            using var faceImage = image.ToFaceImage();
            return viewFace.PredictAge(faceImage);
        }

        /// <summary>
        /// 性别预测（自动裁剪）
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.gender_predictor">gender_predictor.csta</a>
        /// </para>
        /// </summary>
        /// <param name="viewFace"></param>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">关键点坐标<para>通过 <see cref="Mark(FaceLandmarker, {{Image}}, FaceInfo)"/> 获取</para></param>
        /// <returns>性别。<see cref="Gender.Unknown"/> 代表识别失败</returns>
        public static Gender PredictGenderWithCrop(this GenderPredictor viewFace, {{Image}} image, FaceMarkPoint[] points)
        {
            using var faceImage = image.ToFaceImage();
            return viewFace.PredictGenderWithCrop(faceImage, points);
        }

        /// <summary>
        /// 性别预测
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.gender_predictor">gender_predictor.csta</a>
        /// </para>
        /// </summary>
        /// <param name="viewFace"></param>
        /// <param name="image">人脸图像信息</param>
        /// <returns>性别。<see cref="Gender.Unknown"/> 代表识别失败</returns>
        public static Gender PredictGender(this GenderPredictor viewFace, {{Image}} image)
        {
            using var faceImage = image.ToFaceImage();
            return viewFace.PredictGender(faceImage);
        }

        /// <summary>
        /// 眼睛状态检测。
        /// <para>
        /// 眼睛的左右是相对图片内容而言的左右。<br />
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.eye_state">eye_state.csta</a>
        /// </para>
        /// </summary>
        /// <param name="viewFace"></param>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">关键点坐标<para>通过 <see cref="Mark(FaceLandmarker, {{Image}}, FaceInfo)"/> 获取</para></param>
        /// <returns></returns>
        public static EyeStateResult Detect(this EyeStateDetector viewFace, {{Image}} image, FaceMarkPoint[] points)
        {
            using var faceImage = image.ToFaceImage();
            return viewFace.Detect(faceImage, points);
        }

        /// <summary>
        /// 识别 <paramref name="image"/> 中的人脸，并返回可跟踪的人脸信息。
        /// </summary>
        /// <param name="viewFace"></param>
        /// <param name="image">人脸图像信息</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="TrackerConfig"/> 重新检测。</returns>
        public static FaceTrackInfo[] Track(this FaceTracker viewFace, {{Image}} image)
        {
            using var faceImage = image.ToFaceImage();
            return viewFace.Track(faceImage);
        }

        /// <summary>
        /// 戴口罩人脸识别
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="viewFace"></param>
        /// <param name="image"></param>
        /// <param name="info"></param>
        /// <returns></returns>
        public static PlotMaskResult Detect(this MaskDetector viewFace, {{Image}} image, FaceInfo info)
        {
            using var faceImage = image.ToFaceImage();
            return viewFace.Detect(faceImage, info);
        }
    }


    /// <summary>
    /// 异步扩展，对于 CPU 绑定的操作直接使用 <see cref="Task.Run(Action)"/> 进行包装。
    /// <para>参见: <a href="https://docs.microsoft.com/zh-cn/dotnet/standard/async-in-depth#deeper-dive-into-task-and-taskt-for-a-cpu-bound-operation">深入了解绑定 CPU 的操作的任务和 Task&lt;T&gt;</a></para>
    /// </summary>
    public static class AsyncExtensions
    {
        public static Task<FaceInfo[]> DetectAsync(this FaceDetector viewFace, {{Image}} image)
            => Task.Run(() => Extensions.Detect(viewFace, image));

        public static Task<FaceMarkPoint[]> MarkAsync(this FaceLandmarker viewFace, {{Image}} image, FaceInfo info)
            => Task.Run(() => Extensions.Mark(viewFace, image, info));

        public static Task<float[]> ExtractAsync(this FaceRecognizer viewFace, {{Image}} image, FaceMarkPoint[] points)
            => Task.Run(() => Extensions.Extract(viewFace, image, points));

        public static Task<AntiSpoofingResult> PredictAsync(this FaceAntiSpoofing viewFace, {{Image}} image, FaceInfo info, FaceMarkPoint[] points)
            => Task.Run(() => Extensions.Predict(viewFace, image, info, points));

        public static Task<AntiSpoofingResult> PredictVideoAsync(this FaceAntiSpoofing viewFace, {{Image}} image, FaceInfo info, FaceMarkPoint[] points)
            => Task.Run(() => Extensions.PredictVideo(viewFace, image, info, points));

        public static Task<QualityResult> DetectAsync(this FaceQuality viewFace, {{Image}} image, FaceInfo info, FaceMarkPoint[] points, QualityType type)
            => Task.Run(() => Extensions.Detect(viewFace, image, info, points, type));

        public static Task<int> PredictAgeWithCropAsync(this AgePredictor viewFace, {{Image}} image, FaceMarkPoint[] points)
            => Task.Run(() => Extensions.PredictAgeWithCrop(viewFace, image, points));

        public static Task<int> PredictAgeAsync(this AgePredictor viewFace, {{Image}} image)
            => Task.Run(() => Extensions.PredictAge(viewFace, image));

        public static Task<Gender> PredictGenderWithCropAsync(this GenderPredictor viewFace, {{Image}} image, FaceMarkPoint[] points)
            => Task.Run(() => Extensions.PredictGenderWithCrop(viewFace, image, points));

        public static Task<Gender> PredictGenderAsync(this GenderPredictor viewFace, {{Image}} image)
            => Task.Run(() => Extensions.PredictGender(viewFace, image));

        public static Task<EyeStateResult> DetectAsync(this EyeStateDetector viewFace, {{Image}} image, FaceMarkPoint[] points)
            => Task.Run(() => Extensions.Detect(viewFace, image, points));

        public static Task<FaceTrackInfo[]> TrackAsync(this FaceTracker viewFace, {{Image}} image)
            => Task.Run(() => Extensions.Track(viewFace, image));

        public static Task<PlotMaskResult> DetectAsync(this MaskDetector viewFace, {{Image}} image, FaceInfo info)
            => Task.Run(() => Extensions.Detect(viewFace, image, info));

    }
}

""";

            context.AddSource("ViewFaceCoreExtension.g.cs", codeTemplate);
        }
    }

    internal class TestSyntaxReceiver : ISyntaxReceiver
    {
        public string ImageTypeName { get; private set; }
        public List<string> NamespaceNames { get; private set; }

        public bool CanGenerate => !string.IsNullOrEmpty(ImageTypeName) && NamespaceNames != null && NamespaceNames.Count > 0;

        public void OnVisitSyntaxNode(SyntaxNode syntaxNode)
        {
            if (NamespaceNames == null)
            {
                if (syntaxNode is CompilationUnitSyntax namespaceDeclarationSyntax)
                {
                    NamespaceNames = namespaceDeclarationSyntax.Usings.Select(x => ((UsingDirectiveSyntax)x).ToString()).ToList();
                    NamespaceNames.Add("using System;");
                    NamespaceNames.Add("using ViewFaceCore.Models;");
                    NamespaceNames.Add("using System.Threading.Tasks;");
                    NamespaceNames = NamespaceNames.Distinct().ToList();
                }
            }
            if (string.IsNullOrEmpty(ImageTypeName))
            {
                if (syntaxNode is ClassDeclarationSyntax classDeclarationSyntax)
                {
                    foreach (var attribute in classDeclarationSyntax.AttributeLists)
                    {
                        foreach (var attributeSyntax in attribute.Attributes)
                        {
                            if (attributeSyntax.Name.ToFullString() == "ViewFaceCoreImplementation")
                            {
                                foreach (AttributeArgumentSyntax argumentSyntax in attributeSyntax.ArgumentList.Arguments)
                                {
                                    if (argumentSyntax.Expression is TypeOfExpressionSyntax typeOfExpressionSyntax)
                                    {
                                        ImageTypeName = ((IdentifierNameSyntax)typeOfExpressionSyntax.Type).Identifier.ToString();
                                    }
                                }
                            }
                        }
                    }
                }
            }

        }
    }
}
