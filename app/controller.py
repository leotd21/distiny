
from app import ui, comparator
from services import file_manager, data_generator

def run():
    uploaded_file = ui.upload_file()
    if uploaded_file:
        df = file_manager.read_file(uploaded_file)
        ui.display_dataframe("📊 Original Data", df)

        n_rows = ui.select_num_rows(len(df))
        model_type = ui.select_model()

        if ui.generate_button():
            synth_df = data_generator.generate(df, n_rows, model_type)
            ui.display_dataframe("🧬 Synthetic Data", synth_df)
            ui.download_button(synth_df)
            comparator.render_distribution_comparison(df, synth_df)
