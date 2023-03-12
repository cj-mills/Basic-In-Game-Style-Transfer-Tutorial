using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

public class StyleTransfer : MonoBehaviour
{
    [Tooltip("The current camera frame that will be fed to the model")]
    public RenderTexture camerInput;
    [Tooltip("The processed output that will be displayed to the user")]
    public RenderTexture processedOutput;

    [Tooltip("Performs the preprocessing and postprocessing steps")]
    public ComputeShader styleTransferShader;

    [Tooltip("The model asset file that will be used when performing inference")]
    public NNModel modelAsset;

    [Tooltip("The backend used when performing inference")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

    // The compiled model used for performing inference
    private Model m_RuntimeModel;

    // The interface used to execute the neural network
    private IWorker engine;


    // Start is called before the first frame update
    void Start()
    {
        // Compile the model asset into an object oriented representation
        m_RuntimeModel = ModelLoader.Load(modelAsset);
        
        // Create a worker that will execute the model with the selected backend
        engine = WorkerFactory.CreateWorker(workerType, m_RuntimeModel);
    }

    // OnDisable is called when the MonoBehavior becomes disabled or inactive
    private void OnDisable()
    {
        // Release the resources allocated for the inference engine
        engine.Dispose();
    }

    // Update is called once per frame
    void Update()
    {
        // Copy data from cameraInput to an SDR Texture2D
        Texture2D imageTexture = ToTexture2D(camerInput, TextureFormat.RGBA32);
        // Apply preprocessing operations
        Texture2D processedImage = ProcessImage(imageTexture, "ProcessInput");

        // Create a Tensor of shape [1, processedImage.height, processedImage.width, 3]
        Tensor input = new Tensor(processedImage, channels: 3);
        // Remove the processedImage variable
        Destroy(processedImage);

        // Execute neural network with the provided input
        engine.Execute(input);
        // Get the raw model output
        Tensor prediction = engine.PeekOutput();
        // Release GPU resources allocated for the Tensor
        input.Dispose();

        // Create a new HDR RenderTexture to store the model output
        RenderTexture modelOutput = new RenderTexture(processedImage.width, processedImage.height, 24, RenderTextureFormat.ARGBHalf);
        // Remove the imageTexture variable
        Destroy(imageTexture);
        // Copy prediction data to modelOutput
        prediction.ToRenderTexture(modelOutput);
        // Release GPU resources allocated for the Tensor
        prediction.Dispose();

        // Copy data from modelOutput to an HDR Texture2D
        imageTexture = ToTexture2D(modelOutput, TextureFormat.RGBAHalf);
        // Remove the modelOutput variable
        Destroy(modelOutput);
        // Apply postprocessing operations
        processedImage = ProcessImage(imageTexture, "ProcessOutput");
        // Remove the imageTexture variable
        Destroy(imageTexture);

        // Copy the data from the Texture2D to the RenderTexture
        Graphics.Blit(processedImage, processedOutput);
        // Remove the processedImage variable
        Destroy(processedImage);
    }

    /// <summary>
    /// Copy the data from a RenderTexture to a new Texture2D with the specified format
    /// </summary>
    /// <param name="rTex"></param>
    /// <param name="format"></param>
    /// <returns>The new Texture2D</returns>
    public static Texture2D ToTexture2D(RenderTexture rTex, TextureFormat format)
    {
        // Create a new Texture2D with the same dimensions as the RenderTexture
        Texture2D dest = new Texture2D(rTex.width, rTex.height, format, false);
        // Copy the RenderTexture contents to the new Texture2D
        Graphics.CopyTexture(rTex, dest);

        return dest;
    }

    /// <summary>
    /// Process the provided image using the specified function on the GPU
    /// </summary>
    /// <param name="image"></param>
    /// <param name="functionName"></param>
    /// <returns>The processed image</returns>
    private Texture2D ProcessImage(Texture2D image, string functionName)
    {
        // Specify the number of threads on the GPU
        int numthreads = 8;
        // Get the index for the PreprocessResNet function in the ComputeShader
        int kernelHandle = styleTransferShader.FindKernel(functionName);
        // Define an HDR RenderTexture
        RenderTexture rTex = new RenderTexture(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
        // Enable random write access
        rTex.enableRandomWrite = true;
        // Create the HDR RenderTexture
        rTex.Create();

        // Set the value for the Result variable in the ComputeShader
        styleTransferShader.SetTexture(kernelHandle, "Result", rTex);
        // Set the value for the InputImage variable in the ComputeShader
        styleTransferShader.SetTexture(kernelHandle, "InputImage", image);

        // Execute the ComputeShader
        styleTransferShader.Dispatch(kernelHandle, image.width / numthreads, image.height / numthreads, 1);
        // Make the HDR RenderTexture the active RenderTexture
        RenderTexture.active = rTex;

        // Create a new HDR Texture2D
        Texture2D nTex = new Texture2D(rTex.width, rTex.height, TextureFormat.RGBAHalf, false);

        // Copy the RenderTexture to the new Texture2D
        Graphics.CopyTexture(rTex, nTex);
        // Make the HDR RenderTexture not the active RenderTexture
        RenderTexture.active = null;
        // Remove the HDR RenderTexture
        Destroy(rTex);
        return nTex;
    }
}
