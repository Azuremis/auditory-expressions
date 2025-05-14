"""
Main entry point for Audio Week 5 project.
Provides a command-line interface to run different audio tasks.
"""

import argparse
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function serving as entry point for all tasks"""
    parser = argparse.ArgumentParser(
        description="Audio Week 5 - Audio processing and machine learning tasks"
    )
    
    # Add main command argument
    parser.add_argument(
        'command',
        choices=['train', 'infer', 'download', 'eval', 'demo'],
        help='Command to execute'
    )
    
    # Add task argument
    parser.add_argument(
        '--task',
        choices=['urbansound', 'whisper', 'music', 'tts'],
        help='Task to run'
    )
    
    # Add config argument
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    # Parse known args to determine which subcommand is being run
    args, remaining = parser.parse_known_args()
    
    if args.command == 'train':
        if args.task == 'urbansound':
            # Import here to avoid loading unnecessary modules
            from src.tasks.train_urbansound import main as train_urbansound
            
            # Create urbansound-specific parser
            urbansound_parser = argparse.ArgumentParser(description="Train UrbanSound8K classifier")
            urbansound_parser.add_argument('--config', type=str, default='configs/urbansound_cnn.yaml',
                                help='Path to configuration file')
            urbansound_parser.add_argument('--output_dir', type=str, default=None,
                                help='Directory to save outputs (default: checkpoints_dir from config)')
            urbansound_parser.add_argument('--use-huggingface', action='store_true',
                                help='Use HuggingFace dataset instead of local files')
            
            # Parse remaining args and call the function
            urbansound_args = urbansound_parser.parse_args(remaining)
            
            # Check if args.config is provided and override the parser's default
            if args.config:
                urbansound_args.config = args.config
                logger.info(f"Using configuration file from main args: {args.config}")
            
            logger.info(f"Training UrbanSound with config: {urbansound_args.config}")
            train_urbansound(urbansound_args)
            
        elif args.task == 'whisper':
            logger.info("Whisper fine-tuning task not yet implemented")
            
        elif args.task == 'music':
            logger.info("Music transcription training task not yet implemented")
            
        elif args.task == 'tts':
            logger.info("TTS training task not yet implemented")
            
        else:
            logger.error(f"Unknown task: {args.task}")
            parser.print_help()
            sys.exit(1)
            
    elif args.command == 'infer':
        if args.task == 'urbansound':
            # Import urbansound inference
            from src.infer.urbansound_inference import main as infer_urbansound
            
            # Create urbansound-specific parser for inference
            urbansound_parser = argparse.ArgumentParser(description="UrbanSound8K Classifier Inference")
            urbansound_parser.add_argument('--model', type=str, required=True,
                                help='Path to model checkpoint file')
            urbansound_parser.add_argument('--input', type=str, default=None,
                                help='Path to audio file or directory containing audio files')
            urbansound_parser.add_argument('--output', type=str, default=None,
                                help='Path to save output visualization (image for single file, directory for batch)')
            urbansound_parser.add_argument('--device', type=str, default=None,
                                help="Device to run inference on ('cuda', 'cpu')")
            urbansound_parser.add_argument('--random-test', action='store_true',
                                help='Use a random test sample from HuggingFace dataset')
            urbansound_parser.add_argument('--test-fold', type=int, default=1,
                                help='Fold to use for random test sample (1-10)')
            
            # Parse remaining args and call the function
            urbansound_args = urbansound_parser.parse_args(remaining)
            infer_urbansound(urbansound_args)
            
        elif args.task == 'whisper':
            logger.info("Whisper inference task not yet implemented")
            
        elif args.task == 'music':
            logger.info("Music transcription inference task not yet implemented")
            
        elif args.task == 'tts':
            logger.info("TTS inference task not yet implemented")
            
        else:
            logger.error(f"Unknown task: {args.task}")
            parser.print_help()
            sys.exit(1)
            
    elif args.command == 'demo':
        if args.task == 'urbansound':
            logger.info("Running UrbanSound8K demo...")
            # Import here to avoid loading unnecessary modules
            from src.infer.urbansound_inference import main as infer_urbansound
            
            # Create a simple parser for the demo
            demo_parser = argparse.ArgumentParser(description="UrbanSound8K Demo")
            demo_parser.add_argument('--model', type=str, default=None,
                                help='Path to model checkpoint file (if not provided, try to find the latest one)')
            demo_parser.add_argument('--device', type=str, default=None,
                                help="Device to run inference on ('cuda', 'cpu')")
            
            # Parse remaining args
            demo_args = demo_parser.parse_args(remaining)
            
            # Find the latest model if not provided
            if demo_args.model is None:
                checkpoints_dir = os.path.join("checkpoints", "urbansound")
                if os.path.exists(checkpoints_dir):
                    model_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")]
                    if model_files:
                        # Sort by modification time (newest first)
                        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
                        demo_args.model = os.path.join(checkpoints_dir, model_files[0])
                        logger.info(f"Using latest model: {demo_args.model}")
                    else:
                        logger.error("No model checkpoints found in checkpoints/urbansound directory")
                        logger.info("Please train a model first or provide a model path")
                        sys.exit(1)
                else:
                    logger.error("Checkpoints directory not found")
                    logger.info("Please train a model first or provide a model path")
                    sys.exit(1)
            
            # Create inference args
            from argparse import Namespace
            infer_args = Namespace(
                model=demo_args.model,
                input=None,
                output=None,
                device=demo_args.device,
                random_test=True,
                test_fold=1
            )
            
            # Run inference with the random test sample
            infer_urbansound(infer_args)
            
    elif args.command == 'download':
        if args.task == 'urbansound':
            logger.info("Downloading UrbanSound8K from HuggingFace...")
            try:
                from datasets import load_dataset
                
                # Create download parser
                download_parser = argparse.ArgumentParser(description="Download UrbanSound8K dataset")
                download_parser.add_argument('--cache-dir', type=str, default=None,
                                help='Directory to store the downloaded dataset')
                
                # Parse remaining args
                download_args = download_parser.parse_args(remaining)
                
                # Download the dataset
                dataset = load_dataset("danavery/urbansound8K", cache_dir=download_args.cache_dir)
                
                logger.info(f"Dataset downloaded successfully with {len(dataset['train'])} examples")
                logger.info("To use this dataset, set 'use_huggingface: true' in your config file")
                
            except Exception as e:
                logger.error(f"Error downloading dataset: {e}")
                logger.info("Alternatively, you can download UrbanSound8K manually from:")
                logger.info("https://urbansounddataset.weebly.com/urbansound8k.html")
                logger.info("Extract it to data/urbansound8k/")
                sys.exit(1)
            
        elif args.task == 'whisper':
            logger.info("Whisper dataset download task not yet implemented")
            
        elif args.task == 'music':
            logger.info("MusicNet download task not yet implemented")
            
        elif args.task == 'tts':
            logger.info("LJSpeech download task not yet implemented")
            
        else:
            logger.error(f"Unknown task: {args.task}")
            parser.print_help()
            sys.exit(1)
            
    elif args.command == 'eval':
        logger.info(f"Evaluation for {args.task} not yet implemented")
        
    else:
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
