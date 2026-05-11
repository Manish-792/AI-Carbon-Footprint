"""
Streamlit Dashboard for Carbon Footprint AI Framework
Interactive visualization and analysis of AI model carbon emissions
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.config import (
    RESULTS_DIR, MODELS_DIR, PLOTS_DIR, FIGURES_DIR, TABLES_DIR,
    CARBON_INTENSITY, COLOR_PALETTE, ALL_MODELS
)
from scripts.metrics import MetricsProcessor, calculate_green_ai_score

# Page configuration
st.set_page_config(
    page_title="Carbon Footprint AI Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .green-text { color: #2A9D8F; }
    .red-text { color: #E63946; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_results() -> Dict[str, Any]:
    """Load experiment results from JSON files"""
    results = {}
    
    # Load main results
    main_results_path = RESULTS_DIR / "experiment_results.json"
    if main_results_path.exists():
        with open(main_results_path) as f:
            results["main"] = json.load(f)
    
    # Load individual model results
    results["models"] = {}
    for result_file in RESULTS_DIR.glob("*_result.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)
                model_name = data.get("model_name", result_file.stem)
                results["models"][model_name] = data
        except Exception:
            pass
    
    # Load metrics
    metrics_file = RESULTS_DIR / "all_model_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            results["metrics"] = json.load(f)
    
    return results


@st.cache_data
def load_metrics_processor() -> MetricsProcessor:
    """Load metrics processor with results"""
    processor = MetricsProcessor()
    processor.load_from_results(RESULTS_DIR)
    return processor


def get_sample_data() -> pd.DataFrame:
    """Generate sample data if no results exist"""
    return pd.DataFrame({
        "Model": ["SimpleCNN", "ResNet-18", "MobileNetV2", "ResNet-18-Pruned", "ResNet-18-Quantized"],
        "Accuracy (%)": [75.5, 92.3, 88.7, 90.1, 91.8],
        "Parameters (M)": [0.06, 11.2, 2.2, 5.6, 11.2],
        "Size (MB)": [0.24, 42.7, 8.5, 21.4, 10.7],
        "Training Time (min)": [5.0, 30.0, 20.0, 10.0, 0.5],
        "Energy (Wh)": [12.5, 75.0, 50.0, 25.0, 5.0],
        "CO2 (g)": [10.3, 61.5, 41.0, 20.5, 4.1],
        "Green AI Score": [7327, 1501, 2163, 4395, 22390],
        "Optimization": ["None", "None", "None", "Pruned", "Quantized"]
    })


# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar():
    """Render sidebar with navigation and settings"""
    st.sidebar.title("🌱 Carbon AI Dashboard")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Overview", "📈 Model Comparison", "🧮 Carbon Calculator", 
         "🔧 Optimization Lab", "🏆 Leaderboard", "🌍 Regional Analysis",
         "📥 Export"]
    )
    
    st.sidebar.markdown("---")
    
    # Settings
    st.sidebar.subheader("Settings")
    region = st.sidebar.selectbox(
        "Region",
        list(CARBON_INTENSITY.keys()),
        index=0,
        format_func=lambda x: f"{x.title()} ({CARBON_INTENSITY[x]} kgCO₂/kWh)"
    )
    
    st.sidebar.markdown("---")
    
    # Info
    st.sidebar.info("""
    **About**
    
    This dashboard visualizes the carbon footprint 
    of AI model training and helps identify 
    energy-efficient models.
    
    **Green AI Score** = Accuracy / CO₂
    
    Higher is better!
    """)
    
    return page, region


# ============================================================================
# PAGES
# ============================================================================
def render_overview(data: pd.DataFrame, results: Dict):
    """Render overview page"""
    st.markdown('<h1 class="main-header">🌱 Carbon Footprint of AI Models</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    > **Research Framework for Measuring and Reducing AI Carbon Emissions**
    
    This dashboard provides comprehensive analysis of energy consumption and carbon 
    emissions from training deep learning models, with a focus on optimization strategies.
    """)
    
    # Key metrics
    st.subheader("📊 Key Findings")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Models Analyzed",
            len(data),
            help="Total number of models in the comparison"
        )
    
    with col2:
        best_acc = data["Accuracy (%)"].max()
        best_model = data.loc[data["Accuracy (%)"].idxmax(), "Model"]
        st.metric(
            "Best Accuracy",
            f"{best_acc:.1f}%",
            help=f"Achieved by {best_model}"
        )
    
    with col3:
        total_co2 = data["CO2 (g)"].sum()
        st.metric(
            "Total CO₂ Emitted",
            f"{total_co2:.1f}g",
            help="Combined emissions from all experiments"
        )
    
    with col4:
        best_green = data["Green AI Score"].max()
        greenest = data.loc[data["Green AI Score"].idxmax(), "Model"]
        st.metric(
            "Best Green AI Score",
            f"{best_green:.0f}",
            help=f"Most efficient: {greenest}"
        )
    
    st.markdown("---")
    
    # Quick visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Accuracy vs CO₂ Emissions")
        fig = px.scatter(
            data,
            x="CO2 (g)",
            y="Accuracy (%)",
            size="Parameters (M)",
            color="Optimization",
            hover_name="Model",
            title="Model Trade-off Analysis",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            xaxis_title="CO₂ Emissions (g)",
            yaxis_title="Accuracy (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Green AI Score Ranking")
        sorted_data = data.sort_values("Green AI Score", ascending=True)
        fig = px.bar(
            sorted_data,
            x="Green AI Score",
            y="Model",
            orientation='h',
            color="Green AI Score",
            color_continuous_scale="Greens",
            title="Higher is Better"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary table
    st.subheader("📋 Model Summary")
    st.dataframe(
        data.style.background_gradient(subset=["Accuracy (%)", "Green AI Score"], cmap="Greens")
              .background_gradient(subset=["CO2 (g)", "Energy (Wh)"], cmap="Reds_r"),
        use_container_width=True
    )


def render_model_comparison(data: pd.DataFrame, region: str):
    """Render model comparison page"""
    st.header("📈 Model Comparison")
    
    # Model selection
    selected_models = st.multiselect(
        "Select models to compare",
        data["Model"].tolist(),
        default=data["Model"].tolist()[:4]
    )
    
    if not selected_models:
        st.warning("Please select at least one model")
        return
    
    filtered_data = data[data["Model"].isin(selected_models)]
    
    # Comparison metrics
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Performance", "⚡ Energy", "🌍 Carbon", "📐 Size"]
    )
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                filtered_data,
                x="Model",
                y="Accuracy (%)",
                color="Model",
                title="Accuracy Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                filtered_data,
                x="Model",
                y="Training Time (min)",
                color="Model",
                title="Training Time"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.bar(
            filtered_data,
            x="Model",
            y="Energy (Wh)",
            color="Model",
            title="Energy Consumption (Wh)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Calculate regional emissions
        carbon_intensity = CARBON_INTENSITY[region]
        regional_co2 = filtered_data["Energy (Wh)"].values / 1000 * carbon_intensity * 1000
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=filtered_data["Model"],
            y=regional_co2,
            marker_color='#E63946',
            name=f'CO₂ ({region})'
        ))
        fig.update_layout(
            title=f"CO₂ Emissions in {region.title()} ({carbon_intensity} kgCO₂/kWh)",
            yaxis_title="CO₂ (g)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                filtered_data,
                x="Model",
                y="Parameters (M)",
                color="Model",
                title="Parameter Count (Millions)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                filtered_data,
                x="Model",
                y="Size (MB)",
                color="Model",
                title="Model Size (MB)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart comparison
    st.subheader("🎯 Multi-Metric Comparison")
    
    if len(selected_models) >= 2:
        # Normalize data for radar
        metrics = ["Accuracy (%)", "Green AI Score", "Parameters (M)", "Energy (Wh)", "CO2 (g)"]
        
        fig = go.Figure()
        
        for _, row in filtered_data.iterrows():
            # Normalize values (higher is better except for energy/co2/params)
            max_vals = filtered_data[metrics].max()
            normalized = [
                row["Accuracy (%)"] / max_vals["Accuracy (%)"],
                row["Green AI Score"] / max_vals["Green AI Score"],
                1 - row["Parameters (M)"] / max_vals["Parameters (M)"],  # Invert
                1 - row["Energy (Wh)"] / max_vals["Energy (Wh)"],  # Invert
                1 - row["CO2 (g)"] / max_vals["CO2 (g)"]  # Invert
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=normalized,
                theta=["Accuracy", "Green Score", "Compactness", "Energy Eff.", "Carbon Eff."],
                fill='toself',
                name=row["Model"]
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Normalized Multi-Metric Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_carbon_calculator(region: str):
    """Render carbon calculator page"""
    st.header("🧮 Carbon Footprint Calculator")
    
    st.markdown("""
    Estimate the carbon footprint of your AI model training before running experiments.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Parameters")
        
        num_params = st.number_input(
            "Number of parameters (millions)",
            min_value=0.01,
            max_value=1000.0,
            value=10.0,
            step=0.1
        )
        
        epochs = st.slider("Training epochs", 1, 100, 20)
        
        batch_size = st.selectbox(
            "Batch size",
            [16, 32, 64, 128, 256],
            index=2
        )
        
        dataset_size = st.number_input(
            "Dataset size (samples)",
            min_value=1000,
            max_value=10000000,
            value=50000,
            step=1000
        )
    
    with col2:
        st.subheader("Hardware & Region")
        
        gpu_power = st.slider(
            "GPU power consumption (W)",
            50, 400, 150,
            help="RTX 3090: ~350W, RTX 4050: ~115W"
        )
        
        selected_region = st.selectbox(
            "Training region",
            list(CARBON_INTENSITY.keys()),
            index=list(CARBON_INTENSITY.keys()).index(region),
            format_func=lambda x: f"{x.title()}"
        )
        
        carbon_intensity = CARBON_INTENSITY[selected_region]
        st.info(f"Carbon intensity: {carbon_intensity} kgCO₂/kWh")
    
    # Calculate estimates
    if st.button("Calculate Carbon Footprint", type="primary"):
        # Rough estimation
        params_m = num_params * 1e6
        batches_per_epoch = dataset_size // batch_size
        time_per_batch = params_m * 1e-9 * batch_size * 3  # forward + backward + update
        estimated_time_s = epochs * batches_per_epoch * time_per_batch * 1.2
        
        energy_kwh = (gpu_power * estimated_time_s) / (1000 * 3600)
        co2_kg = energy_kwh * carbon_intensity
        
        st.markdown("---")
        st.subheader("📊 Estimated Carbon Footprint")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Time", f"{estimated_time_s/60:.1f} min")
        
        with col2:
            st.metric("Energy Consumption", f"{energy_kwh*1000:.2f} Wh")
        
        with col3:
            st.metric("CO₂ Emissions", f"{co2_kg*1000:.2f} g")
        
        # Regional comparison
        st.subheader("🌍 Regional Comparison")
        
        regional_data = []
        for reg, intensity in CARBON_INTENSITY.items():
            regional_data.append({
                "Region": reg.title(),
                "CO₂ (g)": energy_kwh * intensity * 1000,
                "Intensity": intensity
            })
        
        df = pd.DataFrame(regional_data)
        fig = px.bar(
            df,
            x="Region",
            y="CO₂ (g)",
            color="Intensity",
            color_continuous_scale="RdYlGn_r",
            title="Same Training in Different Regions"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tips
        st.subheader("💡 Reduction Tips")
        tips = []
        
        if num_params > 50:
            tips.append("Consider model pruning or knowledge distillation to reduce parameters")
        if epochs > 50:
            tips.append("Use early stopping to reduce unnecessary training")
        if batch_size < 64:
            tips.append("Larger batch sizes can improve GPU utilization")
        if carbon_intensity > 0.5:
            tips.append("Consider training in a region with cleaner energy grid")
        
        for tip in tips:
            st.info(f"💡 {tip}")


def render_optimization_lab(data: pd.DataFrame):
    """Render optimization lab page"""
    st.header("🔧 Optimization Lab")
    
    st.markdown("""
    Compare optimization techniques and their impact on model efficiency.
    """)
    
    # Filter to show optimization comparisons
    baseline_models = data[data["Optimization"] == "None"]
    optimized_models = data[data["Optimization"] != "None"]
    
    if optimized_models.empty:
        st.info("No optimized models found. Run optimization experiments first.")
        
        # Show expected optimizations
        st.subheader("Available Optimization Techniques")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🔪 Pruning
            Remove unnecessary weights to reduce model size.
            
            - **Expected CO₂ reduction**: 30-60%
            - **Accuracy impact**: 1-3% drop
            - **Best for**: Large models
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Quantization
            Reduce precision from FP32 to INT8.
            
            - **Expected size reduction**: 2-4x
            - **Inference speedup**: 2-3x
            - **Accuracy impact**: <1% drop
            """)
        
        with col3:
            st.markdown("""
            ### 🎓 Distillation
            Transfer knowledge to smaller model.
            
            - **Expected CO₂ reduction**: 40-70%
            - **Size reduction**: 3-10x
            - **Accuracy retention**: 90-98%
            """)
        
        return
    
    # Optimization comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            data,
            x="Model",
            y="CO2 (g)",
            color="Optimization",
            barmode="group",
            title="CO₂ Emissions by Optimization"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            data,
            x="Model",
            y="Accuracy (%)",
            color="Optimization",
            barmode="group",
            title="Accuracy by Optimization"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Savings analysis
    st.subheader("💰 Optimization Savings")
    
    for _, opt_row in optimized_models.iterrows():
        opt_type = opt_row["Optimization"]
        
        # Find baseline
        base_arch = opt_row["Model"].split("-")[0]
        baseline = baseline_models[baseline_models["Model"].str.contains(base_arch)]
        
        if not baseline.empty:
            base_row = baseline.iloc[0]
            
            co2_savings = (base_row["CO2 (g)"] - opt_row["CO2 (g)"]) / base_row["CO2 (g)"] * 100
            acc_change = opt_row["Accuracy (%)"] - base_row["Accuracy (%)"]
            
            with st.expander(f"{opt_row['Model']} ({opt_type})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("CO₂ Savings", f"{co2_savings:.1f}%")
                with col2:
                    st.metric("Accuracy Change", f"{acc_change:+.1f}%")
                with col3:
                    st.metric("Green AI Score", f"{opt_row['Green AI Score']:.0f}")


def render_leaderboard(data: pd.DataFrame):
    """Render leaderboard page"""
    st.header("🏆 Green AI Leaderboard")
    
    metric = st.selectbox(
        "Rank by",
        ["Green AI Score", "Accuracy (%)", "Energy (Wh)", "CO2 (g)"]
    )
    
    ascending = metric in ["Energy (Wh)", "CO2 (g)"]
    sorted_data = data.sort_values(metric, ascending=ascending).reset_index(drop=True)
    sorted_data.index = sorted_data.index + 1
    sorted_data.index.name = "Rank"
    
    # Podium
    if len(sorted_data) >= 3:
        st.subheader("🥇 Top 3")
        
        col1, col2, col3 = st.columns(3)
        
        with col2:  # Gold in middle
            st.markdown("### 🥇 1st Place")
            st.markdown(f"**{sorted_data.iloc[0]['Model']}**")
            st.metric(metric, f"{sorted_data.iloc[0][metric]:.2f}")
        
        with col1:  # Silver on left
            st.markdown("### 🥈 2nd Place")
            st.markdown(f"**{sorted_data.iloc[1]['Model']}**")
            st.metric(metric, f"{sorted_data.iloc[1][metric]:.2f}")
        
        with col3:  # Bronze on right
            st.markdown("### 🥉 3rd Place")
            st.markdown(f"**{sorted_data.iloc[2]['Model']}**")
            st.metric(metric, f"{sorted_data.iloc[2][metric]:.2f}")
    
    st.markdown("---")
    
    # Full leaderboard
    st.subheader("📋 Full Rankings")
    
    display_cols = ["Model", metric, "Accuracy (%)", "CO2 (g)", "Optimization"]
    st.dataframe(
        sorted_data[display_cols].style.background_gradient(subset=[metric], cmap="Greens" if not ascending else "Greens_r"),
        use_container_width=True
    )


def render_regional_analysis(data: pd.DataFrame):
    """Render regional analysis page"""
    st.header("🌍 Regional Carbon Analysis")
    
    st.markdown("""
    The same model training produces different carbon emissions depending on where 
    it's run, due to varying grid carbon intensities.
    """)
    
    # Regional comparison for selected model
    selected_model = st.selectbox("Select model", data["Model"].tolist())
    model_data = data[data["Model"] == selected_model].iloc[0]
    
    # Calculate regional emissions
    energy_kwh = model_data["Energy (Wh)"] / 1000
    
    regional_data = []
    for region, intensity in CARBON_INTENSITY.items():
        regional_data.append({
            "Region": region.title(),
            "CO₂ (g)": energy_kwh * intensity * 1000,
            "Carbon Intensity": intensity
        })
    
    df = pd.DataFrame(regional_data)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            df,
            x="Region",
            y="CO₂ (g)",
            color="Carbon Intensity",
            color_continuous_scale="RdYlGn_r",
            title=f"CO₂ Emissions for {selected_model}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            df,
            values="Carbon Intensity",
            names="Region",
            title="Carbon Intensity by Region",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Savings potential
    st.subheader("💡 Potential Savings")
    
    max_co2 = df["CO₂ (g)"].max()
    min_co2 = df["CO₂ (g)"].min()
    savings = max_co2 - min_co2
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Highest Emissions", f"{max_co2:.2f}g", 
                 help=df.loc[df["CO₂ (g)"].idxmax(), "Region"])
    with col2:
        st.metric("Lowest Emissions", f"{min_co2:.2f}g",
                 help=df.loc[df["CO₂ (g)"].idxmin(), "Region"])
    with col3:
        st.metric("Potential Savings", f"{savings:.2f}g ({savings/max_co2*100:.1f}%)")
    
    # All models regional comparison
    st.subheader("📊 All Models - Regional Comparison")
    
    all_regional = []
    for _, row in data.iterrows():
        energy = row["Energy (Wh)"] / 1000
        for region, intensity in CARBON_INTENSITY.items():
            all_regional.append({
                "Model": row["Model"],
                "Region": region.title(),
                "CO₂ (g)": energy * intensity * 1000
            })
    
    df_all = pd.DataFrame(all_regional)
    
    fig = px.bar(
        df_all,
        x="Model",
        y="CO₂ (g)",
        color="Region",
        barmode="group",
        title="Regional Emissions Comparison"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def render_export(data: pd.DataFrame):
    """Render export page"""
    st.header("📥 Export Results")
    
    st.markdown("Download your analysis results and visualizations.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Export")
        
        # CSV export
        csv = data.to_csv(index=False)
        st.download_button(
            "📄 Download CSV",
            csv,
            "carbon_footprint_results.csv",
            "text/csv",
            use_container_width=True
        )
        
        # JSON export
        json_data = data.to_json(orient="records", indent=2)
        st.download_button(
            "📋 Download JSON",
            json_data,
            "carbon_footprint_results.json",
            "application/json",
            use_container_width=True
        )
    
    with col2:
        st.subheader("📈 Available Figures")
        
        # List available figures
        if FIGURES_DIR.exists():
            figures = list(FIGURES_DIR.glob("*.png"))
            if figures:
                for fig_path in figures:
                    st.text(f"✅ {fig_path.name}")
            else:
                st.info("No figures generated yet. Run experiments first.")
        else:
            st.info("Figures directory not found.")
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    
    stats = {
        "Total Models": len(data),
        "Avg Accuracy (%)": data["Accuracy (%)"].mean(),
        "Avg CO₂ (g)": data["CO2 (g)"].mean(),
        "Total Energy (Wh)": data["Energy (Wh)"].sum(),
        "Total CO₂ (g)": data["CO2 (g)"].sum(),
        "Best Green AI Score": data["Green AI Score"].max()
    }
    
    stats_df = pd.DataFrame([stats])
    st.dataframe(stats_df, use_container_width=True)


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Sidebar
    page, region = render_sidebar()
    
    # Load data
    results = load_results()
    
    # Get data (use sample if no results)
    if results.get("models"):
        processor = load_metrics_processor()
        if processor.models:
            data = processor.get_comparison_table()
        else:
            data = get_sample_data()
            st.warning("Using sample data. Run experiments to see real results.")
    else:
        data = get_sample_data()
        st.warning("No experiment results found. Using sample data for demonstration.")
    
    # Render selected page
    if page == "📊 Overview":
        render_overview(data, results)
    elif page == "📈 Model Comparison":
        render_model_comparison(data, region)
    elif page == "🧮 Carbon Calculator":
        render_carbon_calculator(region)
    elif page == "🔧 Optimization Lab":
        render_optimization_lab(data)
    elif page == "🏆 Leaderboard":
        render_leaderboard(data)
    elif page == "🌍 Regional Analysis":
        render_regional_analysis(data)
    elif page == "📥 Export":
        render_export(data)


if __name__ == "__main__":
    main()
