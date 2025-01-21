import json
import pathlib
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

class AbstractGenerator:
    def __init__(self, data_dir: str = "arxiv_samples", output_dir: str = "generated_samples"):
        self.data_dir = pathlib.Path(data_dir)
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.client = OpenAI()
        
    def load_2021_papers(self) -> List[dict]:
        """Load papers from 2021 data file."""
        target_file = self.data_dir / "papers_2021.json"
        if not target_file.exists():
            raise FileNotFoundError(f"2021 papers file not found at {target_file}")
            
        with open(target_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['papers']
            
    def generate_abstract(self, title: str, original_abstract: str) -> str:
        """Generate a new abstract using GPT-4-mini."""
        try:
            prompt = f"""As a research paper writer, rewrite the following academic abstract in a different way 
            while maintaining the same key findings and technical accuracy. Keep the style formal and academic.
            
            Title: {title}
            Original Abstract: {original_abstract}
            
            Generated Abstract:"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",  # Replace with actual GPT-4-mini model identifier
                messages=[
                    {"role": "system", "content": "You are an expert academic writer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating abstract: {e}")
            return None
            
    def process_papers(self, papers: List[dict], sample_size: int = None) -> List[dict]:
        """Process papers and generate new abstracts."""
        if sample_size:
            import random
            papers = random.sample(papers, sample_size)
            
        generated_papers = []
        
        for paper in tqdm(papers, desc="Generating abstracts"):
            generated_abstract = self.generate_abstract(paper['title'], paper['abstract'])
            
            if generated_abstract:
                generated_papers.append({
                    'title': paper['title'],
                    'original_abstract': paper['abstract'],
                    'generated_abstract': generated_abstract,
                    'original_id': paper.get('id', ''),
                    'created': paper.get('created', ''),
                    'categories': paper.get('categories', [])
                })
                
        return generated_papers
        
    def save_results(self, generated_papers: List[dict], filename: str = "generated_abstracts_2021.json"):
        """Save generated abstracts to JSON file."""
        output_path = self.output_dir / filename
        
        output_data = {
            'year': 2021,
            'count': len(generated_papers),
            'papers': generated_papers
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"Saved {len(generated_papers)} generated abstracts to {output_path}")
        
        # Also save as CSV for easy viewing
        df = pd.DataFrame(generated_papers)
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV version to {csv_path}")

def main():
    generator = AbstractGenerator()
    
    print("Loading 2021 papers...")
    papers = generator.load_2021_papers()
    
    print(f"Loaded {len(papers)} papers from 2021")
    
    # You can specify a smaller sample size for testing
    sample_size = 100  # Change this as needed
    print(f"Processing {sample_size} papers...")
    
    generated_papers = generator.process_papers(papers, sample_size=sample_size)
    generator.save_results(generated_papers)

if __name__ == "__main__":
    main()