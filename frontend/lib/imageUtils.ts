/**
 * Image preprocessing utilities for frontend
 * Optimizes images before uploading to backend
 */

/**
 * Resize and compress image before upload
 * @param file Original image file
 * @param maxDimension Maximum width or height (default 1920px)
 * @param quality JPEG quality 0-1 (default 0.9)
 * @returns Processed image file
 */
export async function preprocessImage(
  file: File,
  maxDimension: number = 1920,
  quality: number = 0.9
): Promise<File> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      reject(new Error('Canvas context not available'));
      return;
    }

    img.onload = () => {
      let width = img.width;
      let height = img.height;

      // Calculate new dimensions maintaining aspect ratio
      if (width > maxDimension || height > maxDimension) {
        if (width > height) {
          height = Math.round((height * maxDimension) / width);
          width = maxDimension;
        } else {
          width = Math.round((width * maxDimension) / height);
          height = maxDimension;
        }
      }

      canvas.width = width;
      canvas.height = height;

      // Draw image with high quality
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(img, 0, 0, width, height);

      // Convert to Blob
      canvas.toBlob(
        (blob) => {
          if (!blob) {
            reject(new Error('Image conversion failed'));
            return;
          }

          // Create new File from Blob
          const newFile = new File([blob], file.name, {
            type: 'image/jpeg',
            lastModified: Date.now(),
          });

          resolve(newFile);
        },
        'image/jpeg',
        quality
      );
    };

    img.onerror = () => {
      reject(new Error('Failed to load image'));
    };

    img.src = URL.createObjectURL(file);
  });
}

/**
 * Get original dimensions of an image file
 * @param file Image file
 * @returns Width and height
 */
export function getImageDimensions(file: File): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      resolve({ width: img.width, height: img.height });
      URL.revokeObjectURL(img.src);
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

/**
 * Validate image file before processing
 * @param file Image file to validate
 * @returns Error message if invalid, null if valid
 */
export function validateImageFile(file: File): string | null {
  // Check file type
  if (!file.type.startsWith('image/')) {
    return 'Please select an image file';
  }

  // Check file size (10MB limit - reduced for Vercel/serverless compatibility)
  const maxSizeMB = 10;
  const fileSizeMB = file.size / (1024 * 1024);
  if (fileSizeMB > maxSizeMB) {
    return `File too large (${fileSizeMB.toFixed(1)}MB). Maximum size is ${maxSizeMB}MB. Please compress or resize your image.`;
  }
  
  // Additional check: prevent extremely large files that could cause buffer issues
  const maxSizeBytes = 10 * 1024 * 1024; // 10MB in bytes
  if (file.size > maxSizeBytes) {
    return `File size exceeds ${maxSizeMB}MB limit. Please use a smaller image.`;
  }

  return null;
}

/**
 * Validate image dimensions (minimum 400x400px)
 * @param file Image file to validate
 * @returns Error message if invalid, null if valid
 */
export async function validateImageDimensions(file: File): Promise<string | null> {
  try {
    const { width, height } = await getImageDimensions(file);

    const minDimension = 400;
    if (width < minDimension || height < minDimension) {
      return `Image too small (${width}×${height}px). Minimum ${minDimension}×${minDimension}px required for quality detection and embeddings.`;
    }

    return null;
  } catch (error) {
    return 'Failed to read image dimensions. Please try another file.';
  }
}

/**
 * Calculate file size reduction percentage
 * @param originalSize Original file size in bytes
 * @param newSize New file size in bytes
 * @returns Reduction percentage
 */
export function calculateReduction(originalSize: number, newSize: number): number {
  return Math.round(((originalSize - newSize) / originalSize) * 100);
}

