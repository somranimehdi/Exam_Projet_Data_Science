import React, { useCallback } from "react";
import { Upload, ImagePlus } from "lucide-react";
import { useDropzone } from "react-dropzone";
import { motion } from "framer-motion";

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
}

export function ImageUpload({ onImageSelect }: ImageUploadProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onImageSelect(acceptedFiles[0]);
      }
    },
    [onImageSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".jpeg", ".jpg", ".png"],
    },
    multiple: false,
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      {...getRootProps()}
      className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all
        ${
          isDragActive
            ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
            : "border-gray-300 dark:border-gray-600 hover:border-blue-400 dark:hover:border-blue-500"
        }
        dark:bg-gray-800`}
    >
      <input {...getInputProps()} />
      <motion.div
        className="flex flex-col items-center gap-4"
        animate={{ scale: isDragActive ? 1.1 : 1 }}
      >
        {isDragActive ? (
          <Upload className="w-12 h-12 text-blue-500" />
        ) : (
          <ImagePlus className="w-12 h-12 text-gray-400 dark:text-gray-500" />
        )}
        <div className="space-y-2">
          <p className="text-lg font-medium text-gray-900 dark:text-gray-100">
            {isDragActive ? "Drop the image here" : "Drag & drop an image here"}
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            or click to select a file
          </p>
        </div>
      </motion.div>
    </motion.div>
  );
}
