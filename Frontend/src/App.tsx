import React, { useState } from "react";
import { ImageUpload } from "./components/ImageUpload";
import { ImagePreview } from "./components/ImagePreview";
import { ResultCard } from "./components/ResultCard";
import { LoadingSpinner } from "./components/LoadingSpinner";
import { ThemeToggle } from "./components/ThemeToggle";
import { Camera } from "lucide-react";
import { useTheme } from "./hooks/useTheme";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";

const mockPrediction = async (image: File) => {
  // Create a FormData object to hold the image file
  const formData = new FormData();
  formData.append("file", image);

  try {
    // Send the file to the FastAPI endpoint using Axios
    const response = await axios.post(
      "http://localhost:8000/uploadfile/",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );
    let results = [response.data];
    return results;
  } catch (error) {
    console.error("Error sending file to backend:", error);
    throw new Error("Failed to process the image.");
  }
};
function App() {
  const { theme, toggleTheme } = useTheme();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<
    Array<{ label: string; confidence: number }>
  >([]);

  const handleImageSelect = async (file: File) => {
    const imageUrl = URL.createObjectURL(file);
    setSelectedImage(imageUrl);
    setIsLoading(true);
    setResults([]);

    try {
      const predictions = await mockPrediction(file);
      setResults(predictions);
    } catch (error) {
      console.error("Error processing image:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRemoveImage = () => {
    setSelectedImage(null);
    setResults([]);
  };

  return (
    <div
      className={`min-h-screen transition-colors duration-200
      ${theme === "dark" ? "bg-gray-900" : "bg-gray-50"}`}
    >
      <ThemeToggle theme={theme} toggleTheme={toggleTheme} />

      <div className="max-w-3xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <Camera className="mx-auto h-12 w-12 text-blue-500" />
          <motion.h1
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="mt-3 text-3xl font-bold text-gray-900 dark:text-white"
          >
            Traffic Sign Detection
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="mt-2 text-gray-600 dark:text-gray-300"
          >
            Upload an image to identify traffic signs and signals
          </motion.p>
        </motion.div>

        <div className="space-y-8">
          <AnimatePresence mode="wait">
            {!selectedImage ? (
              <ImageUpload key="upload" onImageSelect={handleImageSelect} />
            ) : (
              <ImagePreview
                key="preview"
                imageUrl={selectedImage}
                onRemove={handleRemoveImage}
              />
            )}
          </AnimatePresence>

          <AnimatePresence>{isLoading && <LoadingSpinner />}</AnimatePresence>

          <AnimatePresence>
            {results.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-4"
              >
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  Detection Results
                </h2>
                <div className="grid gap-4">
                  {results.map((result, index) => (
                    <ResultCard key={index} result={result} index={index} />
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}

export default App;
