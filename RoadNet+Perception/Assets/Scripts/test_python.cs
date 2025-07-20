using UnityEngine;
using Python.Runtime;
using System;
using System.IO;
using System.Collections;
using UnityEngine.Rendering;


public class test_python : MonoBehaviour
{
    public enum Models
    {
        best_model_fast
    };

    [Header("Frame Capturing Settings")]
    public Camera mainCamera;
    public int width = 600, height = 600;
    public float initialFrameCapturingCooldown = 1f;
    public float frameCapturingInterval = 0.25f;
    private Coroutine _frameAnalysisCoroutine;
    private RenderTexture _rt;
    private CommandBuffer _cameraCB;

    [Header("Python Segmentation Model\'s Settings")]
    public Models selectedModel;
    //private PyObject _pythonModule;
    private PyObject _inferenceModel;
    private PyObject _inferenceFunction;

    void Awake()
    {
        string pyDir = Path.Combine(Application.streamingAssetsPath, "python");
        string modelPath = Path.Combine(pyDir, "inference", selectedModel.ToString() + ".pth");

        using (Py.GIL())
        {
            PyObject pythonModule;

            using (PyObject sys = Py.Import("sys"))
            {
                sys.GetAttr("path")
                    .InvokeMethod("append", new PyTuple(new[] { new PyString(pyDir) }));

                using PyObject modules = sys.GetAttr("modules");
                using PyObject key = new PyString("inference.inference_core");
                using PyObject popped = modules.InvokeMethod(
                        "pop", new PyTuple(new[] { key, PyObject.None }));

                pythonModule = Py.Import("inference.inference_core");
            }

            // Load selected Inference model
            using PyObject constructor = pythonModule.GetAttr("DrivingSegmentationInference");
            using PyObject pyModelPath = new PyString(modelPath);
            _inferenceModel = constructor.Invoke(new PyObject[] { pyModelPath });

            // Get Inference Function
            _inferenceFunction = _inferenceModel.GetAttr("predict_bytes");

            pythonModule.Dispose();
        }
    }

    public void OnDestroy()
    {
        // CameraBuffer
        mainCamera.RemoveCommandBuffer(CameraEvent.AfterEverything, _cameraCB);
        _rt.Release();
        Destroy(_rt);

        // Embedded Python's "Garbage Collector"
        _inferenceModel.Dispose();

        if (PythonEngine.IsInitialized)
            PythonEngine.Shutdown();
    }

    void Start()
    {
        // Setup RenderTexture object for frame capturing subroutine
        _rt = RenderTexture.GetTemporary(width, height, 24, RenderTextureFormat.ARGB32);
        _rt.Create();

        // Setup Camera Command Buffer in order to avoid GPU sincronization
        // and camera jittering during gameplay by using unity's own render pipeline
        _cameraCB = new CommandBuffer { name = "CaptureFrameCB" };
        _cameraCB.Blit(BuiltinRenderTextureType.CameraTarget, _rt);
        mainCamera.AddCommandBuffer(CameraEvent.AfterEverything, _cameraCB);

        _frameAnalysisCoroutine = StartCoroutine(
            CaptureFrameLoop()
        );
    }

    void AnalizeFrame(AsyncGPUReadbackRequest req)
    {
        if (req.hasError)
        {
            Debug.LogError("Error en GPU Readback");
            return;
        }

        byte[] data = req.GetData<byte>().ToArray();

        using (Py.GIL())
        {
            // Convierte los bytes de C# a PyBytes
            using PyObject pyBytes = PyObject.FromManagedObject(data);
            // Llama a la función Python
            PyObject pyResult = _inferenceFunction.Invoke(pyBytes);

            // Convierte el resultado (también bytes) de nuevo a byte[]
            byte[] resultBytes = pyResult.As<byte[]>();

            Debug.Log(resultBytes[0]);
        }
    }

    IEnumerator CaptureFrameLoop()
    {
        yield return new WaitForSeconds(initialFrameCapturingCooldown);

        while (true)
        {
            yield return new WaitForSeconds(frameCapturingInterval);
            // Also tried this w/ WaitForEndOfFrame, but it had weird behaviour in the compiled project
            yield return null; // Wait for the next frame to end (for rendering)

            // Send an asynchronous request to the RenderTexture associated with the
            // Camera buffer in order to take a picture of the current frame
            AsyncGPUReadback.Request(_rt, 0, TextureFormat.RGBA32, AnalizeFrame);
        }
    }
}
