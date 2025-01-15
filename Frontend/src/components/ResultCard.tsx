import React from 'react';
import { AlertCircle, CheckCircle2 } from 'lucide-react';
import { motion } from 'framer-motion';

interface DetectionResult {
  label: string;
  confidence: number;
}

interface ResultCardProps {
  result: DetectionResult;
  index: number;
}

export function ResultCard({ result, index }: ResultCardProps) {
  const isHighConfidence = result.confidence >= 0.7;

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.1 }}
      className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 flex items-center gap-4
        hover:shadow-lg transition-shadow"
    >
      <motion.div
        whileHover={{ rotate: 360 }}
        transition={{ duration: 0.5 }}
      >
        {isHighConfidence ? (
          <CheckCircle2 className="w-6 h-6 text-green-500" />
        ) : (
          <AlertCircle className="w-6 h-6 text-yellow-500" />
        )}
      </motion.div>
      <div>
        <h3 className="font-medium text-gray-900 dark:text-gray-100">{result.label}</h3>
        <div className="mt-1 w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${result.confidence * 100}%` }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
            className={`h-full rounded-full ${
              isHighConfidence ? 'bg-green-500' : 'bg-yellow-500'
            }`}
          />
        </div>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Confidence: {(result.confidence * 100).toFixed(1)}%
        </p>
      </div>
    </motion.div>
  );
}