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
        driving_segmentation_fast,
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
                using PyObject key = new PyString("inference.inference_fast");
                using PyObject popped = modules.InvokeMethod(
                        "pop", new PyTuple(new[] { key, PyObject.None }));

                pythonModule = Py.Import("inference.inference_fast");
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
    }

    void Start()
    {
        // Setup RenderTexture object for frame capturing subroutine
        _rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
        _rt.Create();

        _frameAnalysisCoroutine = StartCoroutine(
            CaptureFrameLoop()
        );
    }

    void OnReadback(AsyncGPUReadbackRequest req)
    {
        if (req.hasError)
        {
            Debug.LogError("Error en GPU Readback");
            return;
        }

        Color32[] rawPixels = req.GetData<Color32>().ToArray();
        int w = _rt.width;
        int h = _rt.height;

        // 1) Flip vertical: crea un nuevo array y copia fila por fila invertida
        Color32[] flipped = new Color32[rawPixels.Length];
        for (int y = 0; y < h; y++)
        {
            int srcRow = y * w;
            int dstRow = (h - 1 - y) * w;
            Array.Copy(rawPixels, srcRow, flipped, dstRow, w);
        }

        // 2) Crea y llena la textura en CPU con los píxeles ya volteados
        var tmp = new Texture2D(w, h, TextureFormat.RGBA32, false);
        tmp.SetPixels32(flipped);
        tmp.Apply();

        // 3) Codifica la textura a PNG
        byte[] pngBytes = tmp.EncodeToPNG();
        Destroy(tmp);

        try
        {
            using (Py.GIL())
            {
                using var pyBytes = PyObject.FromManagedObject(pngBytes);
                _inferenceFunction.Invoke(pyBytes);
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error en inferencia Python: {ex}");
        }
    }

    IEnumerator CaptureFrameLoop()
    {
        yield return new WaitForSeconds(initialFrameCapturingCooldown);

        while (true)
        {
            yield return new WaitForSeconds(frameCapturingInterval);
            yield return new WaitForEndOfFrame(); // Cambiado de null a WaitForEndOfFrame
            
            Debug.Log("Iniciando captura de frame...");

            Graphics.Blit(null, _rt);

            // Pide la lectura asíncrona directamente sobre _rt
            AsyncGPUReadback.Request(_rt, 0, TextureFormat.RGBA32, OnReadback);
        }
    }
}
