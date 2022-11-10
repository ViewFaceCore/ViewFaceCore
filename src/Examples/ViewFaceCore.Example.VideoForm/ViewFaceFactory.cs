using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ViewFaceCore.Configs;
using ViewFaceCore.Configs.Enums;
using ViewFaceCore.Core;
using ViewFaceCore.Models;

namespace ViewFaceCore.Demo.VideoForm
{
    internal class ViewFaceFactory : IDisposable
    {
        public int Width;
        public int Height;

        public ViewFaceFactory(int width, int height)
        {
            this.Width = width;
            this.Height = height;
        }

        MaskDetector _maskDetector = null;
        FaceDetector _faceDetector = null;
        FaceLandmarker _faceMark = null;
        //FaceType需要用口罩模型
        FaceRecognizer _faceRecognizer = null;
        //FaceType需要用口罩模型
        FaceRecognizer _maskFaceRecognizer = null;

        AgePredictor _agePredictor = null;
        GenderPredictor _genderPredictor = null;
        //人脸追踪
        FaceTracker _faceTracker = null;

        public T Get<T>() where T : class
        {
            switch (typeof(T).Name)
            {
                case nameof(MaskDetector):
                    {
                        if (_maskDetector == null)
                            _maskDetector = new MaskDetector();
                        return _maskDetector as T;
                    }
                case nameof(FaceDetector):
                    {
                        if (_faceDetector == null)
                            _faceDetector = new FaceDetector(new FaceDetectConfig() { MaxWidth = this.Width, MaxHeight = this.Height });
                        return _faceDetector as T;
                    }
                case nameof(FaceLandmarker):
                    {
                        if (_faceMark == null)
                            _faceMark = new FaceLandmarker();
                        return _faceMark as T;
                    }
                case nameof(FaceRecognizer):
                    {
                        if (_faceRecognizer == null)
                        {
                            var config = new FaceRecognizeConfig(FaceType.Normal)
                            {
                                Threshold = 0.7f
                            };
                            _faceRecognizer = new FaceRecognizer();
                        }
                        return _faceRecognizer as T;
                    }
                case nameof(AgePredictor):
                    {
                        if (_agePredictor == null)
                            _agePredictor = new AgePredictor();
                        return _agePredictor as T;
                    }
                case nameof(GenderPredictor):
                    {
                        if (_genderPredictor == null)
                            _genderPredictor = new GenderPredictor();
                        return _genderPredictor as T;
                    }
                case nameof(FaceTracker):
                    {
                        if (_faceTracker == null)
                            _faceTracker = new FaceTracker(new FaceTrackerConfig(this.Width, this.Height)
                            {
                                MinFaceSize = 20,
                                Stable = true,
                            });
                        return _faceTracker as T;
                    }
                default:
                    {
                        throw new Exception($"Can not get type {typeof(T)}");
                    }
            }
        }



        public FaceRecognizer GetFaceRecognizerWithMask()
        {
            if (_maskFaceRecognizer == null)
            {
                var config = new FaceRecognizeConfig(FaceType.Mask);
                _maskFaceRecognizer = new FaceRecognizer();
            }
            return _maskFaceRecognizer;
        }

        public void Dispose()
        {
            if (_maskDetector != null) _maskDetector.Dispose();
            if (_faceDetector != null) _faceDetector.Dispose();
            if (_faceMark != null) _faceMark.Dispose();
            if (_faceRecognizer != null) _faceRecognizer.Dispose();
            if (_agePredictor != null) _agePredictor.Dispose();
            if (_genderPredictor != null) _genderPredictor.Dispose();
            if (_maskFaceRecognizer != null) _maskFaceRecognizer.Dispose();
            if (_faceTracker != null) _faceTracker.Dispose();
        }
    }
}
