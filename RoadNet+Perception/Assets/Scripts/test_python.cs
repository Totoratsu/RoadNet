using UnityEngine;
using Python.Runtime;
using System;
using System.IO;
using System.Collections;
using UnityEngine.Rendering;


public class test_python : MonoBehaviour
{
    [Header("Frame Capturing Settings")]
    public Camera camara;
    public int width = 600, height = 600;
    public float InitialFrameCapturingCooldown = 1f;
    public float FrameCapturingInterval = 0.25f;
    private Coroutine _frameAnalysisCoroutine;

    private PyObject funcion;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Awake()
    {
        //Runtime.PythonDLL = Path.Combine(Application.streamingAssetsPath, "embedded-python", "python312.dll");
        //PythonEngine.Initialize(); com.unity.scripting.python

        using (Py.GIL())
        {
            // 2) Añade StreamingAssets/python a sys.path
            using (PyObject sys = Py.Import("sys"))
            {
                string pyDir = Path.Combine(Application.streamingAssetsPath, "python");
                // equivale a: sys.path.append(pyDir)
                sys
                  .GetAttr("path")
                  .InvokeMethod("append", new PyTuple(new[] { new PyString(pyDir) }));

                // 3) (Opcional) Limpia el cache de un módulo antiguo
                using (PyObject modules = sys.GetAttr("modules"))
                using (PyObject key = new PyString("test"))
                using (PyObject def = PyObject.None)
                using (PyObject popped = modules.InvokeMethod("pop", new PyTuple(new[] { key, def })))
                {
                    // aquí popped es el módulo viejo o None
                }
            }

            // 4) Importa por primera vez tu script
            funcion = Py.Import("test").GetAttr("guardar_imagen_desde_string");
        }
    }

    public void OnDestroy()
    {
        // Limpia las referencias
        funcion.Dispose();

        if (PythonEngine.IsInitialized)
            PythonEngine.Shutdown();
    }

    void Start()
    {
        _frameAnalysisCoroutine = StartCoroutine(
            CaptureFrameRoutine(camara, width, height)
        );
    }

    void AnalizeFrame(AsyncGPUReadbackRequest req)
    {
        if (req.hasError) {
            Debug.LogError("Error en GPU Readback");
            return;
        }

        byte[] data = req.GetData<byte>().ToArray();
    }

    IEnumerator CaptureFrameRoutine(Camera cam, int width, int height) {
        yield return new WaitForSeconds(InitialFrameCapturingCooldown);

        while (true)
        {
            yield return new WaitForSeconds(FrameCapturingInterval);
            // Also tried this w/ WaitForEndOfFrame, but it had weird behaviour in the compiled project
            yield return null; // Wait for the next frame to end (for rendering)

            var rt = new RenderTexture(width, height, 24);
            cam.targetTexture = rt;
            cam.Render();
            cam.targetTexture = null;

            AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32, AnalizeFrame);

            RenderTexture.ReleaseTemporary(rt);
        }
    }
}
