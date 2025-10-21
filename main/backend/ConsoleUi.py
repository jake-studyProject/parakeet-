import yaml

import time
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.console import Console

import AsrWorker as aw

class ConsoleUi:
    def __init__(self,asr_worker:aw,cfg_file):
        self.asr_worker = asr_worker

        #Console attributes 
        self.console = Console()
        self.running = True
        self.refresh_rate = cfg_file["refresh_rate"]
        
        self.title = cfg_file["title"]
        self.text_style = cfg_file["text_style"]
        self.border_style = cfg_file["border_style"]
    

    def run(self):
        with Live(refresh_per_second=self.refresh_rate,console=self.console) as live:
            while self.running:
                stable = "\n".join(f"[{ts}] {speaker}: {text}" for ts,speaker,text in self.asr_worker.final_segments)
                partial = self.asr_worker.current_partial

                
                
                txt = Text()
                txt.append(stable)

                if partial:
                    txt.append("\n" + str(partial), style=self.text_style)
                panel = Panel(
                    txt,
                    title=self.title,
                    border_style=self.border_style,
                )
                live.update(panel)
                time.sleep(0.05)

