using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Python.Runtime;
using System;
using System.IO;

public class test : MonoBehaviour
{

    public float speed = 0.1f;

    private PyObject _thisObject;
    private PyObject _moverModule;
    private PyObject _oscillateFunc;

    private void Awake()
    {
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
                using (PyObject key = new PyString("mover"))
                using (PyObject def = PyObject.None)
                using (PyObject popped = modules.InvokeMethod("pop", new PyTuple(new[] { key, def })))
                {
                    // aquí popped es el módulo viejo o None
                }
            }

            // 4) Importa por primera vez tu script
            _moverModule = Py.Import("mover");
            _oscillateFunc = _moverModule.GetAttr("oscillate");
            
            _thisObject = PyObject.FromManagedObject(this.gameObject);
        }

    }

    public void OnDestroy()
    {
        // Limpia las referencias
        _oscillateFunc.Dispose();
        _moverModule.Dispose();
        _thisObject.Dispose();

        if (PythonEngine.IsInitialized)
            PythonEngine.Shutdown();
    }
    
    void Update()
    {
        using (Py.GIL())
        {
            // Convierte los parámetros a PyObject
            var pyStep = new PyFloat(speed);

            // Llama a oscillate(go, step)
            using PyObject result = _oscillateFunc.Invoke(new PyObject[] { _thisObject, pyStep });

            float num = result.As<float>();
            Debug.Log(num);

            // Limpia las referencias temporales
            pyStep.Dispose();
        }
    }
}
