import click
import task.ppu.main as PPU
# if __name__=="__main__":
#     PPU.run()

@click.command()
@click.option('-v', '--verbose',required = False, is_flag=True, default=True, help="Display verbose infomation")
@click.option('-m', '--module',required = True, help="Use which network, resnet or alexnet.")
@click.option('-g', '--gpu',required = False, is_flag=True, default=False, help="Use gpu")
@click.option('-h', '--half',required = False, is_flag=True, default=False, help="Use fp16")
@click.option('-l', '--load_params',required = False, default=None, help="Load params")
def ppu_data(verbose, module, gpu, half, load_params):
    """Generate test data of ppu.
    """
    if module=="resnet":
        load_params = "model/resnet18_baseline_new_archi3_bnAffineTrue_fp16/model_best.pth.tar"
    PPU.run(verbose=verbose,
            module=module,
            use_gpu=gpu,
            use_half=half,
            load_params=load_params)
    pass
    
@click.group()
def cli():
    pass

cli.add_command(ppu_data)
if __name__=="__main__":
    cli()