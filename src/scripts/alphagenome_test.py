from alphagenome.data import genome
from alphagenome.visualization import plot_components
from alphagenome_research.model import dna_model
import matplotlib.pyplot as plt

if __name__ == "__main__":

    model = dna_model.create("/beegfs/prj/RNA_NLP/AlphaGenome/weights/alphagenome/all_folds/1")

    interval = genome.Interval(chromosome='chr22', start=35677410, end=36725986)
    variant = genome.Variant(
        chromosome='chr22',
        position=36201698,
        reference_bases='A',
        alternate_bases='C',
    )

    outputs = model.predict_variant(
        interval=interval,
        variant=variant,
        ontology_terms=['UBERON:0001157'],
        requested_outputs=[dna_model.OutputType.RNA_SEQ],
    )

    plot_components.plot(
        [
            plot_components.OverlaidTracks(
                tdata={
                    'REF': outputs.reference.rna_seq,
                    'ALT': outputs.alternate.rna_seq,
                },
                colors={'REF': 'dimgrey', 'ALT': 'red'},
            ),
        ],
        interval=outputs.reference.rna_seq.interval.resize(2**15),
        # Annotate the location of the variant as a vertical line.
        annotations=[plot_components.VariantAnnotation([variant], alpha=0.8)],
    )
    plt.savefig("/beegfs/prj/pratikum_ws_2025/project1/results/alphagenome_test.png")