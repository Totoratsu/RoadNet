using UnityEngine;
using Python.Runtime;
using System;
using System.IO;
using System.Collections;  


public class test_python : MonoBehaviour
{
    public Camera camara;

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

    // Update is called once per frame
    void Start()
    {
        using (Py.GIL())
        {
            StartCoroutine(
                CaptureFrameAsBase64Coroutine(camara, 600, 600, base64Str =>
                {
                    using (PyObject pyStr = new PyString(base64Str))
                    using (PyObject name = new PyString("hola.png"))
                    using (funcion.Invoke(new PyObject[] { pyStr, name }));
                })
            );
        }
    }
    
    IEnumerator CaptureFrameAsBase64Coroutine(Camera cam, int width, int height, Action<string> callback) {
        var rt = new RenderTexture(width, height, 24);
        cam.targetTexture = rt;
        cam.Render();

        RenderTexture.active = rt;
        var tex = new Texture2D(width, height, TextureFormat.ARGB32, false);
        tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        tex.Apply();

        cam.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);

        byte[] png = tex.EncodeToPNG();
        Destroy(tex);

        callback(Convert.ToBase64String(png));
        yield return null;
    }
}
