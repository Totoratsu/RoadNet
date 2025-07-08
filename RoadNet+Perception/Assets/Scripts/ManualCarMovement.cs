using UnityEngine;
using UnityEngine.InputSystem;

public class ManualCarMovement : MonoBehaviour
{
    public InputActionAsset UserInputActions;
    private Rigidbody carRigidbody;

    private InputAction moveAction;
    private Vector2 moveAmount;
    private InputAction rotateAction;
    private Vector2 rotateAmount;
    private InputAction changeCamera;
    private float toggleCamera;

    [Header("Car Control")]
    public float thrust = 1000f;
    public float brakeForce = 1000f;
    public float rotAngle = 7.5f;

    [Header("Camera Control")]
    public GameObject firstPersonCamera;
    public GameObject thirdPersonCamera;

    private void OnEnable()
    {
        // Empezamos a escuchar las acciones del action map "Player"
        UserInputActions.FindActionMap("Player").Enable();
    }

    private void OnDisable()
    {
        // Dejamos de escuchar las acciones del action map cuando
        // este objeto es deshabilitado o eliminado
        UserInputActions.FindActionMap("Player").Disable();
    }

    private void Awake()
    {
        moveAction = UserInputActions.FindAction("Gas");
        rotateAction = UserInputActions.FindAction("Rotate");
        changeCamera = UserInputActions.FindAction("SwitchCam");

        carRigidbody = GetComponent<Rigidbody>();
    }

    public void FixedUpdate()
    {
        moveAmount = moveAction.ReadValue<Vector2>();
        rotateAmount = rotateAction.ReadValue<Vector2>();


        if (moveAmount.y > 0) // Hit Gas
        {
            carRigidbody.AddForce(
                transform.forward * thrust * Time.deltaTime,
                ForceMode.Force
            );
        }
        else if (moveAmount.y < 0) // Brake
        {
            carRigidbody.AddForce(
                -transform.forward * brakeForce * Time.deltaTime,
                ForceMode.Force
            );
        }

        if (rotateAmount.x != 0)
        {
            transform.Rotate(
                0f,
                rotateAmount.x * rotAngle * Time.fixedDeltaTime,
                0f
            );
        }
    }

    private void Update()
    {
        toggleCamera = changeCamera.ReadValue<float>();

        if (changeCamera.WasPressedThisFrame())
        {
            if (toggleCamera < 0)
            {
                thirdPersonCamera.SetActive(false);
                firstPersonCamera.SetActive(true);
            }
            else
            {
                thirdPersonCamera.SetActive(true);
                firstPersonCamera.SetActive(false);
            }
        }
    }

}
