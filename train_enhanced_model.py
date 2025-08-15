#!/usr/bin/env python3
"""
CLI-Skript für das Training der erweiterten Gesichtserkennung mit Metadaten
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# App-Imports
from app.enhanced_face_engine import MetadataAwareTrainer, EnhancedFaceEngine

def main():
    parser = argparse.ArgumentParser(
        description="Trainieren Sie die erweiterte Gesichtserkennung mit Metadaten"
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Pfad zu JSON-Dateien mit Trainingsdaten"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="models/enhanced_model.pkl",
        help="Ausgabe-Pfad für das trainierte Modell"
    )
    
    parser.add_argument(
        "--validation-split", "-v",
        type=float,
        default=0.2,
        help="Anteil der Daten für Validierung (0.1-0.5)"
    )
    
    parser.add_argument(
        "--age-weight",
        type=float,
        default=0.3,
        help="Gewichtung für Alters-Erkennung"
    )
    
    parser.add_argument(
        "--gender-weight",
        type=float,
        default=0.25,
        help="Gewichtung für Geschlechts-Erkennung"
    )
    
    parser.add_argument(
        "--location-weight",
        type=float,
        default=0.2,
        help="Gewichtung für Standort-Metadaten"
    )
    
    parser.add_argument(
        "--temporal-weight",
        type=float,
        default=0.15,
        help="Gewichtung für zeitliche Metadaten"
    )
    
    parser.add_argument(
        "--technical-weight",
        type=float,
        default=0.1,
        help="Gewichtung für technische Metadaten"
    )
    
    parser.add_argument(
        "--verbose", "-V",
        action="store_true",
        help="Ausführliche Ausgabe"
    )
    
    args = parser.parse_args()
    
    # Validierung
    if not os.path.exists(args.input):
        print(f"❌ Eingabe-Pfad existiert nicht: {args.input}")
        sys.exit(1)
    
    if not (0.1 <= args.validation_split <= 0.5):
        print("❌ Validierungs-Split muss zwischen 0.1 und 0.5 liegen")
        sys.exit(1)
    
    # Ausgabe-Verzeichnis erstellen
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starte Training der erweiterten Gesichtserkennung...")
    print(f"📁 Eingabe: {args.input}")
    print(f"💾 Ausgabe: {args.output}")
    print(f"📊 Validierungs-Split: {args.validation_split}")
    
    try:
        # Trainingsdaten laden
        training_data = load_training_data(args.input)
        
        if not training_data:
            print("❌ Keine Trainingsdaten gefunden!")
            sys.exit(1)
        
        print(f"✅ {len(training_data)} Trainingsbeispiele geladen")
        
        # Metadaten-Gewichtungen
        metadata_weights = {
            'age': args.age_weight,
            'gender': args.gender_weight,
            'location': args.location_weight,
            'temporal': args.temporal_weight,
            'technical': args.technical_weight
        }
        
        if args.verbose:
            print("📊 Metadaten-Gewichtungen:")
            for key, value in metadata_weights.items():
                print(f"  {key}: {value}")
        
        # Trainer initialisieren
        trainer = MetadataAwareTrainer(args.output)
        
        # Training durchführen
        results = trainer.train(training_data, args.validation_split)
        
        # Ergebnisse ausgeben
        print("\n" + "="*50)
        print("🎉 TRAINING ERFOLGREICH ABGESCHLOSSEN!")
        print("="*50)
        
        if 'training' in results:
            print("\n📊 Training-Metriken:")
            for metric, value in results['training'].items():
                print(f"  {metric}: {value:.3f}")
        
        if 'validation' in results:
            print("\n✅ Validierungs-Metriken:")
            for metric, value in results['validation'].items():
                print(f"  {metric}: {value:.3f}")
        
        if 'improvement' in results:
            print("\n📈 Verbesserungen:")
            for metric, improvement in results['improvement'].items():
                print(f"  {metric}: {improvement:+.3f}")
        
        print(f"\n💾 Modell gespeichert: {args.output}")
        print("="*50)
        
        # Modell-Info-Datei erstellen
        create_model_info(args.output, results, metadata_weights)
        
    except Exception as e:
        print(f"❌ Fehler beim Training: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def load_training_data(input_path: str) -> List[Dict]:
    """Lädt Trainingsdaten aus JSON-Dateien"""
    training_data = []
    
    if os.path.isfile(input_path):
        # Einzelne Datei
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    training_data.extend(data)
                else:
                    training_data.append(data)
        except Exception as e:
            print(f"⚠️ Fehler beim Laden von {input_path}: {e}")
    
    elif os.path.isdir(input_path):
        # Verzeichnis - suche nach JSON-Dateien
        import glob
        json_files = glob.glob(os.path.join(input_path, "**/*.json"), recursive=True)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        training_data.extend(data)
                    else:
                        training_data.append(data)
            except Exception as e:
                print(f"⚠️ Fehler beim Laden von {json_file}: {e}")
    
    return training_data

def create_model_info(model_path: str, results: Dict, metadata_weights: Dict):
    """Erstellt eine Info-Datei für das trainierte Modell"""
    info_path = model_path.replace('.pkl', '_info.json')
    
    info = {
        'model_path': model_path,
        'created_at': str(Path(model_path).stat().st_mtime),
        'metadata_weights': metadata_weights,
        'training_results': results,
        'model_type': 'enhanced_face_recognition',
        'version': '1.0.0'
    }
    
    try:
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"📄 Modell-Info erstellt: {info_path}")
    except Exception as e:
        print(f"⚠️ Fehler beim Erstellen der Modell-Info: {e}")

def validate_training_data(training_data: List[Dict]) -> Dict[str, Any]:
    """Validiert Trainingsdaten"""
    stats = {
        'total_samples': len(training_data),
        'samples_with_faces': 0,
        'samples_with_metadata': 0,
        'total_faces': 0,
        'age_labels': 0,
        'gender_labels': 0,
        'quality_labels': 0,
        'camera_models': set(),
        'locations': set()
    }
    
    for item in training_data:
        # Gesichter
        persons = item.get('persons', [])
        if persons:
            stats['samples_with_faces'] += 1
            stats['total_faces'] += len(persons)
            
            for person in persons:
                if person.get('age'):
                    stats['age_labels'] += 1
                if person.get('gender'):
                    stats['gender_labels'] += 1
                if person.get('quality_score'):
                    stats['quality_labels'] += 1
        
        # Metadaten
        metadata = item.get('metadata', {})
        if metadata:
            stats['samples_with_metadata'] += 1
            
            if metadata.get('camera_model'):
                stats['camera_models'].add(metadata['camera_model'])
            
            if metadata.get('location', {}).get('country'):
                stats['locations'].add(metadata['location']['country'])
    
    return stats

if __name__ == "__main__":
    main()
