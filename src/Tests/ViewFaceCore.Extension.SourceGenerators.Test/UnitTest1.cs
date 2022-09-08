namespace ViewFaceCore.Extension.SourceGenerators.Test;

using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Text;
using System.Diagnostics;
using System.Reflection;
using System.Text;
using ViewFaceCore.Extension.SourceGenerators;
using VerifyCS = CSharpSourceGeneratorVerifier<ViewFaceCoreImplementationGenerator>;


[TestClass]
public class UnitTest1
{
    [TestMethod]
    public async void TestMethod1()
    {
        var code = "initial code";
        var generated = "expected generated code";
        await new VerifyCS.Test
        {
            TestState =
            {
                Sources = { code },
                GeneratedSources =
                {
                    (typeof(ViewFaceCoreImplementationGenerator), "GeneratedFileName", SourceText.From(generated, Encoding.UTF8, SourceHashAlgorithm.Sha256)),
                },
            },
        }.RunAsync();
    }

    [TestMethod]
    public void SimpleGeneratorTest()
    {
        Compilation inputCompilation = CreateCompilation(@"
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using ViewFaceCore.Attributes;
using ViewFaceCore.Model;

namespace ViewFaceCore.Core
{

    /// <summary>
    /// 
    /// </summary>
    [ViewFaceCoreImplementation(typeof(Bitmap))]
    public static class ViewFaceCoreSystemDrawingExtension
    {

    }
}

");

        ViewFaceCoreImplementationGenerator generator = new ViewFaceCoreImplementationGenerator();
        GeneratorDriver driver = CSharpGeneratorDriver.Create(generator);
        driver = driver.RunGeneratorsAndUpdateCompilation(inputCompilation, out var outputCompilation, out var diagnostics);
        GeneratorDriverRunResult runResult = driver.GetRunResult();
        var code = runResult.GeneratedTrees.FirstOrDefault().ToString();
    }

    private static Compilation CreateCompilation(string source)
        => CSharpCompilation.Create("compilation",
            new[] { CSharpSyntaxTree.ParseText(source) },
            new[] { MetadataReference.CreateFromFile(typeof(Binder).GetTypeInfo().Assembly.Location) },
            new CSharpCompilationOptions(OutputKind.ConsoleApplication));

}