import React from "react";
import { X } from "lucide-react";
import { motion } from "framer-motion";

interface ImagePreviewProps {
  imageUrl: string;
  onRemove: () => void;
}

export function ImagePreview({ imageUrl, onRemove }: ImagePreviewProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      className="relative flex justify-center items-center w-auto h-auto  bg-gray-100 dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden"
    >
      <motion.img
        src={imageUrl}
        alt="Preview"
        className="object-contain w-full h-full"
        layoutId="preview-image"
      />
      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        onClick={onRemove}
        className="absolute top-2 right-2 p-2 bg-white dark:bg-gray-800 rounded-full shadow-md 
          hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
      >
        <X className="w-5 h-5 text-gray-600 dark:text-gray-300" />
      </motion.button>
    </motion.div>
  );
}
