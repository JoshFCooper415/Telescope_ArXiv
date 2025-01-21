import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
import random
import pathlib
import sys
from http.client import IncompleteRead
import backoff
import os
import tarfile
import io
import base64

class ArxivOAIHarvester:
    def __init__(self, base_dir="arxiv_samples_fulltext"):
        self.base_url = "http://export.arxiv.org/oai2?"
        self.base_dir = pathlib.Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        # Create a subdirectory for source files
        self.source_dir = self.base_dir / "sources"
        self.source_dir.mkdir(exist_ok=True)
        self.namespaces = {
            'oai': 'http://www.openarchives.org/OAI/2.0/',
            'arxiv': 'http://arxiv.org/OAI/arXiv/'
        }

    @backoff.on_exception(
        backoff.expo,
        (urllib.error.HTTPError, IncompleteRead),
        max_tries=5
    )
    def fetch_records(self, from_date=None, until_date=None, resumption_token=None):
        """Fetch records using OAI-PMH protocol with proper error handling and backoff."""
        params = {
            'verb': 'ListRecords',
            'metadataPrefix': 'arXiv'  # Using arXiv-specific format for richer metadata
        }
        
        if resumption_token:
            params = {
                'verb': 'ListRecords',
                'resumptionToken': resumption_token
            }
        else:
            if from_date:
                params['from'] = from_date
            if until_date:
                params['until'] = until_date
        
        query = urllib.parse.urlencode(params)
        url = self.base_url + query
        
        try:
            print(f"Fetching records from {url[:100]}...")
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'ArxivHarvester/1.0 (Contact: your@email.com)'}
            )
            response = urllib.request.urlopen(request)
            
            # Handle 503 Retry-After responses
            if response.status == 503:
                retry_after = int(response.headers.get('Retry-After', 30))
                print(f"Server requested delay. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                return self.fetch_records(from_date, until_date, resumption_token)
            
            data = response.read().decode('utf-8')
            time.sleep(5)  # Base delay between requests
            return ET.fromstring(data)
            
        except urllib.error.HTTPError as e:
            if e.code == 503:
                retry_after = int(e.headers.get('Retry-After', 30))
                print(f"Server overloaded. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                return self.fetch_records(from_date, until_date, resumption_token)
            raise

    def download_source(self, arxiv_id):
        """Download and save source files for a given arXiv ID."""
        # Clean the arXiv ID - remove both 'oai:arXiv.' and 'org:' prefixes
        clean_id = arxiv_id.replace('oai:arXiv.', '').replace('org:', '')
        
        source_path = self.source_dir / clean_id
        if source_path.exists():
            return source_path
        
        source_url = f"https://arxiv.org/e-print/{clean_id}"
        
        try:
            print(f"Downloading source for {clean_id}...")
            request = urllib.request.Request(
                source_url,
                headers={'User-Agent': 'ArxivHarvester/1.0 (Contact: your@email.com)'}
            )
            response = urllib.request.urlopen(request)
            
            # Create directory for this paper's sources
            source_path.mkdir(exist_ok=True)
            
            # Read the tar file content
            tar_content = io.BytesIO(response.read())
            
            # Extract all files
            source_files = {}
            with tarfile.open(fileobj=tar_content, mode='r:*') as tar:
                for member in tar.getmembers():
                    if member.isfile():  # Only process files, not directories
                        f = tar.extractfile(member)
                        if f is not None:
                            content = f.read()
                            try:
                                # Try UTF-8 first
                                decoded_content = content.decode('utf-8')
                                encoding = 'utf-8'
                            except UnicodeDecodeError:
                                try:
                                    # Fall back to Latin-1
                                    decoded_content = content.decode('latin-1')
                                    encoding = 'latin-1'
                                except UnicodeDecodeError:
                                    # If text decoding fails, store as base64
                                    decoded_content = base64.b64encode(content).decode('ascii')
                                    encoding = 'base64'
                                    
                            source_files[member.name] = {
                                'content': decoded_content,
                                'size': member.size,
                                'encoding': encoding
                            }
            
            # Be nice to arXiv servers
            time.sleep(3)
            return source_files
            
        except Exception as e:
            print(f"Error downloading source for {clean_id}: {str(e)}")
            return None

    def parse_record(self, record):
        """Parse a single OAI-PMH record and download the source."""
        try:
            metadata = record.find('.//{http://arxiv.org/OAI/arXiv/}arXiv')
            header = record.find('.//{http://www.openarchives.org/OAI/2.0/}header')
            
            if metadata is None or header is None:
                return None
            
            paper = {
                'id': header.find('{http://www.openarchives.org/OAI/2.0/}identifier').text,
                'datestamp': header.find('{http://www.openarchives.org/OAI/2.0/}datestamp').text,
                'title': metadata.find('{http://arxiv.org/OAI/arXiv/}title').text.strip(),
                'abstract': metadata.find('{http://arxiv.org/OAI/arXiv/}abstract').text.strip(),
                'categories': metadata.find('{http://arxiv.org/OAI/arXiv/}categories').text.split(),
                'created': metadata.find('{http://arxiv.org/OAI/arXiv/}created').text,
                'updated': metadata.find('{http://arxiv.org/OAI/arXiv/}updated').text if metadata.find('{http://arxiv.org/OAI/arXiv/}updated') is not None else None,
            }
            
            # Download and store all source files
            source_files = self.download_source(paper['id'])
            if source_files:
                paper['source_files'] = source_files
            
            authors = metadata.findall('.//{http://arxiv.org/OAI/arXiv/}author')
            paper['authors'] = []
            for author in authors:
                keyname = author.find('{http://arxiv.org/OAI/arXiv/}keyname')
                forename = author.find('{http://arxiv.org/OAI/arXiv/}forenames')
                if keyname is not None:
                    author_info = {
                        'keyname': keyname.text,
                        'forename': forename.text if forename is not None else None
                    }
                    paper['authors'].append(author_info)
            
            doi = metadata.find('{http://arxiv.org/OAI/arXiv/}doi')
            if doi is not None:
                paper['doi'] = doi.text
            
            paper['year'] = int(paper['created'][:4])
            
            return paper
            
        except Exception as e:
            print(f"Error parsing record: {str(e)}")
            return None

    def harvest_papers(self, target_years={2024}, papers_per_month=10):
        """Harvest papers evenly distributed across months. Test version with reduced numbers."""
        for year in target_years:
            papers = []
            monthly_counts = {month: 0 for month in range(1, 13)}
            
            print(f"\nHarvesting papers for {year}...")
            print(f"Targeting {papers_per_month} papers per month")
            
            # Single pass for test run
            for month in range(1, 13):
                print(f"\nMonth {month:02d} (Current: {monthly_counts[month]}/{papers_per_month})")
                
                from_date = f"{year}-{month:02d}-01"
                until_date = f"{year}-{month:02d}-31"
                
                resumption_token = None
                attempt = 0
                max_attempts = 3
                
                while (attempt < max_attempts and 
                    monthly_counts[month] < papers_per_month):
                    try:
                        response = self.fetch_records(
                            from_date if not resumption_token else None,
                            until_date if not resumption_token else None,
                            resumption_token
                        )
                        
                        if response is None:
                            break
                        
                        records = response.findall('.//{http://www.openarchives.org/OAI/2.0/}record')
                        if len(records) == 0:
                            break
                        
                        batch_processed = 0
                        
                        for record in records:
                            if monthly_counts[month] >= papers_per_month:
                                break
                                
                            batch_processed += 1
                            paper = self.parse_record(record)
                            
                            if paper is not None:
                                creation_year = int(paper['created'][:4])
                                creation_month = int(paper['created'][5:7])
                                
                                if (creation_year == year and 
                                    creation_month == month):
                                    papers.append(paper)
                                    monthly_counts[month] += 1
                                    print(f"Downloaded paper {monthly_counts[month]}/{papers_per_month} for month {month}")
                        
                        print(f"Processed: {batch_processed}, "
                            f"Month total: {monthly_counts[month]}/{papers_per_month}")
                        
                        if monthly_counts[month] >= papers_per_month:
                            break
                        
                        token_element = response.find('.//{http://www.openarchives.org/OAI/2.0/}resumptionToken')
                        if token_element is None or not token_element.text:
                            break
                        
                        resumption_token = token_element.text
                        print(f"Using resumption token: {resumption_token[:30]}...")
                        
                    except Exception as e:
                        print(f"Error during harvest: {str(e)}")
                        attempt += 1
                        if attempt < max_attempts:
                            print(f"Retrying... (attempt {attempt + 1}/{max_attempts})")
                            time.sleep(30)
                        else:
                            print(f"Failed to harvest papers for {year}-{month} "
                                f"after {max_attempts} attempts")
                            break
            
            # Print monthly distribution
            print(f"\nFinal distribution for {year}:")
            for month in range(1, 13):
                print(f"Month {month:02d}: {monthly_counts[month]} papers")
            
            # Save papers for this year
            output_file = self.base_dir / f"papers_{year}.json"
            data = {
                "year": year,
                "papers": papers
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"\nSaved {len(papers)} papers for {year} to {output_file}")

def main():
        harvester = ArxivOAIHarvester()
        
        print("Starting metadata and source harvest (TEST RUN)...")
        print("This will take some time due to rate limiting.")
        print("Collecting 10 papers per month for current year only")
        
        try:
            # Only harvest current year with 10 papers per month
            harvester.harvest_papers(
                target_years={2024},  # Only current year
                papers_per_month=10   # Just 10 papers per month
            )
            print("\nHarvest complete")
            return 0
        except KeyboardInterrupt:
            print("\nHarvest interrupted by user")
            return 1
        except Exception as e:
            print(f"\nHarvest failed: {str(e)}")
            return 1

if __name__ == "__main__":
    sys.exit(main())