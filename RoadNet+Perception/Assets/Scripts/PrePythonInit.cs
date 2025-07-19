using UnityEngine;
using Python.Runtime;
using System;
using System.IO;

[DefaultExecutionOrder(-100)]
public class PrePythonInit : MonoBehaviour
{
    void Awake()
    {
        Runtime.PythonDLL = Path.Combine(Application.streamingAssetsPath, "embedded-python", "python312.dll");
        PythonEngine.Initialize();
    }
}
