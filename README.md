# **Audio-Query Source Separation with Band-Split Mamba and Latent Diffusion Models**
## **Overview**
This project introduces a novel **audio-query-based source separation** approach, leveraging the **Band-Split Mamba model** and advanced latent diffusion techniques to overcome the limitations of traditional source separation methods.

Conventional approaches are constrained to predefined stems such as **vocals**, **bass**, and **drums**, limiting flexibility and adaptability for more diverse audio queries. This work integrates **timbre** and **pitch features** through a **pre-trained Variational Autoencoder (VAE)** within the **Latent Diffusion Model (LDM)** framework, enabling precise conditioning on the original mixed track.

---

## **Key Features**
- **Audio-Query Source Separation**: Separates user-specified audio stems beyond predefined categories.
- **Band-Split Mamba Model**: Enhances signal processing efficiency with a lightweight architecture.
- **Improved Signal-to-Noise Ratio**: Achieved a **7%+ SNR improvement**, providing cleaner and more accurate separated tracks.
- **Latent Diffusion Framework**: Integrates timbre and pitch conditioning using VAE to capture fine-grained audio features for precise separation.
- **Real-Time Application Focus**: Tackles issues like **inadequate feature extraction** and slow computations, ensuring performance suitable for real-world applications.

---

<!-- ## **Technical Details**
### **1. Band-Split Mamba Model**
- Lightweight and efficient sequence modeling for audio.
- Enhances separation performance by splitting input audio into frequency bands.

### **2. Variational Autoencoder (VAE)**
- Pre-trained VAE extracts and disentangles timbre and pitch features from the audio query.
- Enables conditioning within the LDM framework to refine separation results.

### **3. Latent Diffusion Model (LDM)**
- The LDM framework generates precise reconstructions of audio stems based on user queries.
- Provides high flexibility for handling a wide range of audio sources and combinations.   -->

<!-- --- -->

## **Results**
- **Signal-to-Noise Ratio (SNR):** Achieved a **7%+ improvement** over baseline models.
- Successfully addressed limitations of predefined stems, enabling flexible query-based separation.
- Effective integration of timbre and pitch for **precise source isolation** from mixed audio tracks.

---

<!-- ## **Skills and Tools**
The following technologies and concepts were applied in this project:
- **Machine Learning**: Variational Autoencoders (VAE), Latent Diffusion Models
- **Signal Processing**: Band-Split techniques, SNR optimization
- **Model Frameworks**: Band-Split Mamba for efficient audio processing
- **Domain Expertise**: Digital Signal Processing (DSP), Speech and Source Separation
- **Python**: PyTorch, Librosa, and other audio processing libraries

---

## **Applications**
This project is applicable to a wide range of audio and music-related domains, including:
- **Music Production**: Isolate specific instruments or stems for remixing and mastering.
- **Audio Engineering**: Clean up audio tracks with enhanced SNR for clearer outputs.
- **Live Performance Tools**: Real-time query-based separation for dynamic music generation.

--- -->

<!-- ## **Project Goals**
- Develop flexible **query-based source separation** systems.
- Achieve high-quality separation with minimal computational overhead.
- Enhance real-time performance to bridge the gap between research and real-world use cases.

---

## **Future Directions**
- Optimize the model for faster real-time performance.
- Expand the system to support more complex and hybrid audio queries.
- Explore multi-modal integration with **text-to-audio** and **image-to-audio** query support.

--- -->

## **Author**
**Jeng-Yue Liu**
- Music and Audio Computing Lab, Academia Sinica
- Music and AI Lab, National Taiwan University (NTU)

---

## **Contact**
For further inquiries or collaboration opportunities, please reach out to:
**philip910323@gmail.com**
