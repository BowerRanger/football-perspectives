import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { BrowserRouter } from "react-router"
import { ThemeProvider } from "next-themes"

import "./index.css"
import App from "./App"
import { Toaster } from "@/components/ui/sonner"
import { TooltipProvider } from "@/components/ui/tooltip"
import { DialogsProvider } from "@/hooks/use-dialogs"
import { PipelineProvider } from "@/hooks/use-pipeline"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem storageKey="fp.theme" disableTransitionOnChange>
      <TooltipProvider delayDuration={300}>
        <BrowserRouter>
          <DialogsProvider>
            <PipelineProvider>
              <App />
            </PipelineProvider>
          </DialogsProvider>
        </BrowserRouter>
        <Toaster position="bottom-right" richColors closeButton />
      </TooltipProvider>
    </ThemeProvider>
  </StrictMode>,
)
