#!/usr/bin/env python

import click
import icecream
import inspect

import sci3d as s3d


@click.command()
def main():
    doc = ''
    for attr in dir(s3d):
        attr_ref = getattr(s3d, attr)
        attr_doc = attr_ref.__doc__
        if attr_doc is not None and \
                not attr.startswith('__'):
            doc += f'## sci3d.{attr}{inspect.signature(attr_ref)}\n'
            doc += f'{attr_doc}\n'

    print(doc)


if __name__ == '__main__':
    icecream.install()
    main()
