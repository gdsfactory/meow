import gdsfactory as gf
import numpy as np
import pytest

import meow as mw


class CustomLayers(gf.LayerEnum):
    layout = gf.constant(gf.kcl.layout)
    LAYER1 = (1, 0)
    LAYER2 = (2, 0)


custom_pdk = gf.Pdk(name="custom", layers=CustomLayers)

# To get rid of warnings about compass being unregistered
custom_pdk.register_cells(compass=gf.components.compass)


def test_gdsf_extrusion_builtin_pdk():
    gf.clear_cache()
    gf.gpdk.PDK.activate()

    component = gf.Component()
    component << gf.components.rectangle(size=(1, 1), layer=(1, 0))
    rule = mw.GdsExtrusionRule(material=mw.silicon, h_min=0, h_max=1, mesh_order=1)
    structures = mw.extrude_gds(component, {(1, 0): [rule]})

    assert len(structures) > 0
    assert structures[0].material == mw.silicon
    assert isinstance(structures[0].geometry, mw.Prism)

    poly = structures[0].geometry.poly
    assert np.min(poly[:, 0]) == component.bbox().left
    assert np.min(poly[:, 1]) == component.bbox().bottom
    assert np.max(poly[:, 0]) == component.bbox().right
    assert np.max(poly[:, 1]) == component.bbox().top


def test_gdsf_extrusion_custom_pdk():
    gf.clear_cache()
    custom_pdk.activate()

    component = gf.Component()
    component << gf.components.rectangle(size=(1, 1), layer=(1, 0))
    rule = mw.GdsExtrusionRule(material=mw.silicon, h_min=0, h_max=1, mesh_order=1)
    structures = mw.extrude_gds(component, {(1, 0): [rule]})

    assert len(structures) > 0
    assert structures[0].material == mw.silicon
    assert isinstance(structures[0].geometry, mw.Prism)

    poly = structures[0].geometry.poly
    assert np.min(poly[:, 0]) == component.bbox().left
    assert np.min(poly[:, 1]) == component.bbox().bottom
    assert np.max(poly[:, 0]) == component.bbox().right
    assert np.max(poly[:, 1]) == component.bbox().top


@pytest.mark.parametrize("buffer", [0.1, -0.1])
def test_gdsf_extrusion_buffer(buffer: float):
    gf.clear_cache()
    gf.gpdk.PDK.activate()

    component = gf.Component()
    component << gf.components.rectangle(size=(1, 1), layer=(1, 0))
    extrusion_rules = {
        (1, 0): [
            mw.GdsExtrusionRule(
                material=mw.silicon, h_min=0, h_max=1, mesh_order=1, buffer=buffer
            )
        ]
    }
    structures = mw.extrude_gds(component, extrusion_rules)

    assert len(structures) > 0
    assert structures[0].material == mw.silicon
    geometry: mw.Prism = structures[0].geometry  # ty: ignore[invalid-assignment]
    assert np.isclose(np.min(geometry.poly[:, 0]), component.bbox().left - buffer), (
        f"buffer check failed with {buffer = }"
    )
    assert np.isclose(np.min(geometry.poly[:, 1]), component.bbox().bottom - buffer), (
        f"buffer check failed with {buffer = }"
    )
    assert np.isclose(np.max(geometry.poly[:, 0]), component.bbox().right + buffer), (
        f"buffer check failed with {buffer = }"
    )
    assert np.isclose(np.max(geometry.poly[:, 1]), component.bbox().top + buffer), (
        f"buffer check failed with {buffer = }"
    )


def test_gdsf_extrusion_multiple_rules():
    gf.clear_cache()
    custom_pdk.activate()

    extrusion_rules = {
        (1, 0): [
            mw.GdsExtrusionRule(material=mw.silicon, h_min=0, h_max=1, mesh_order=1)
        ],
        (2, 0): [
            mw.GdsExtrusionRule(
                material=mw.silicon_oxide, h_min=-1, h_max=3, mesh_order=3, buffer=+0.1
            ),
            mw.GdsExtrusionRule(
                material=mw.silicon_nitride, h_min=4, h_max=5, mesh_order=5, buffer=-0.1
            ),
        ],
    }
    component = gf.Component()
    component << gf.components.rectangle(size=(1, 1), layer=(1, 0))
    component << gf.components.rectangle(size=(2, 3), layer=(2, 0))

    structures = mw.extrude_gds(component, extrusion_rules)

    assert len(structures) == 3
    silicon_structure = next(s for s in structures if s.material == mw.silicon)
    assert silicon_structure.mesh_order == 1
    silicon_oxide_structure = next(
        s for s in structures if s.material == mw.silicon_oxide
    )
    assert silicon_oxide_structure.mesh_order == 3
    silicon_nitride_structure = next(
        s for s in structures if s.material == mw.silicon_nitride
    )
    assert silicon_nitride_structure.mesh_order == 5

    si_geometry: mw.Prism = silicon_structure.geometry  # ty: ignore[invalid-assignment]
    sio2_geometry: mw.Prism = silicon_oxide_structure.geometry  # ty: ignore[invalid-assignment]
    sin_geometry: mw.Prism = silicon_nitride_structure.geometry  # ty: ignore[invalid-assignment]

    assert np.isclose(
        np.min(si_geometry.poly[:, 0]),
        component.bbox(layer=int(CustomLayers.LAYER1)).left,  # ty: ignore[invalid-argument-type]
    )
    assert np.isclose(
        np.min(si_geometry.poly[:, 1]),
        component.bbox(layer=int(CustomLayers.LAYER1)).bottom,  # ty: ignore[invalid-argument-type]
    )
    assert np.isclose(
        np.max(si_geometry.poly[:, 0]),
        component.bbox(layer=int(CustomLayers.LAYER1)).right,  # ty: ignore[invalid-argument-type]
    )
    assert np.isclose(
        np.max(si_geometry.poly[:, 1]),
        component.bbox(layer=int(CustomLayers.LAYER1)).top,  # ty: ignore[invalid-argument-type]
    )

    assert np.isclose(
        np.min(sio2_geometry.poly[:, 0]),
        component.bbox(layer=int(CustomLayers.LAYER2)).left - 0.1,  # ty: ignore[invalid-argument-type]
    )
    assert np.isclose(
        np.min(sio2_geometry.poly[:, 1]),
        component.bbox(layer=int(CustomLayers.LAYER2)).bottom - 0.1,  # ty: ignore[invalid-argument-type]
    )
    assert np.isclose(
        np.max(sio2_geometry.poly[:, 0]),
        component.bbox(layer=int(CustomLayers.LAYER2)).right + 0.1,  # ty: ignore[invalid-argument-type]
    )
    assert np.isclose(
        np.max(sio2_geometry.poly[:, 1]),
        component.bbox(layer=int(CustomLayers.LAYER2)).top + 0.1,  # ty: ignore[invalid-argument-type]
    )

    assert np.isclose(
        np.min(sin_geometry.poly[:, 0]),
        component.bbox(layer=int(CustomLayers.LAYER2)).left + 0.1,  # ty: ignore[invalid-argument-type]
    )
    assert np.isclose(
        np.min(sin_geometry.poly[:, 1]),
        component.bbox(layer=int(CustomLayers.LAYER2)).bottom + 0.1,  # ty: ignore[invalid-argument-type]
    )
    assert np.isclose(
        np.max(sin_geometry.poly[:, 0]),
        component.bbox(layer=int(CustomLayers.LAYER2)).right - 0.1,  # ty: ignore[invalid-argument-type]
    )
    assert np.isclose(
        np.max(sin_geometry.poly[:, 1]),
        component.bbox(layer=int(CustomLayers.LAYER2)).top - 0.1,  # ty: ignore[invalid-argument-type]
    )


def test_gdsf_extrusion_two_rectangles():
    gf.clear_cache()
    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_polygon(
        layer=(1, 0),
        points=[(-1, -3), (+1, -3), (+1, -2), (-1, -2)],
    )
    component.add_polygon(
        layer=(1, 0),
        points=[(-1, +3), (+1, +3), (+1, +2), (-1, +2)],
    )

    extrusion_rules = {
        (1, 0): [
            mw.GdsExtrusionRule(
                material=mw.silicon_oxide,
                h_min=0,
                h_max=1,
            )
        ]
    }
    structures = mw.extrude_gds(component, extrusions=extrusion_rules)
    assert len(structures) == 2
    cell = mw.Cell(
        structures=structures,
        z_min=-2,
        z_max=+2,
        mesh=mw.Mesh2D(x=np.linspace(-3, 3, 31), y=np.linspace(-3, 3, 31)),
    )
    structure_left = (
        cell.structures_2d[0]
        if cell.structures_2d[0].geometry.x_min < cell.structures_2d[1].geometry.x_min
        else cell.structures_2d[1]
    )
    structure_right = (
        cell.structures_2d[1]
        if structure_left is cell.structures_2d[0]
        else cell.structures_2d[0]
    )
    left_geometry: mw.Rectangle = structure_left.geometry
    right_geometry: mw.Rectangle = structure_right.geometry
    assert np.isclose(left_geometry.x_min, -3)
    assert np.isclose(left_geometry.x_max, -2)
    assert np.isclose(left_geometry.y_min, 0)
    assert np.isclose(left_geometry.y_max, 1)
    assert np.isclose(right_geometry.x_min, 2)
    assert np.isclose(right_geometry.x_max, 3)
    assert np.isclose(right_geometry.y_min, 0)
    assert np.isclose(right_geometry.y_max, 1)


def test_gdsf_hole():
    gf.clear_cache()
    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_polygon(
        layer=(1, 0),
        points=[
            (5, -5),
            (5, 5),
            (-5, 5),
            (-5, -5),
            (-3, -3),
            (-3, 3),
            (3, 3),
            (3, -3),
            (-3, -3),
            (-5, -5),
        ],
    )

    extrusion_rules = {
        (1, 0): [
            mw.GdsExtrusionRule(
                material=mw.silicon_oxide,
                h_min=0,
                h_max=1,
                buffer=0.5,
            ),
            mw.GdsExtrusionRule(
                material=mw.silicon_nitride,
                h_min=1,
                h_max=2,
                buffer=0,
            ),
            mw.GdsExtrusionRule(
                material=mw.silicon,
                h_min=2,
                h_max=3,
                buffer=-0.5,
            ),
        ]
    }
    structures = mw.extrude_gds(component, extrusions=extrusion_rules)
    cell = mw.Cell(
        structures=structures,
        z_min=-5,
        z_max=5,
        mesh=mw.Mesh2D(x=np.linspace(-6, 6, 51), y=np.linspace(-6, 6, 51)),
    )

    assert (
        len(cell.structures_2d) == 6
    )  # 3 extrusion rules, each makes 2 structures on each side

    silicon_oxide_structures = [
        s for s in cell.structures_2d if s.material == mw.silicon_oxide
    ]
    silicon_nitride_structures = [
        s for s in cell.structures_2d if s.material == mw.silicon_nitride
    ]
    silicon_structures = [s for s in cell.structures_2d if s.material == mw.silicon]

    for structures, buffer, h_min in [
        (silicon_oxide_structures, 0.5, 0),
        (silicon_nitride_structures, 0, 1),
        (silicon_structures, -0.5, 2),
    ]:
        assert (
            len(structures) == 2
        )  # each extrusion rule makes 2 structures on each side
        structure_left = (
            structures[0]
            if structures[0].geometry.x_min < structures[1].geometry.x_min
            else structures[1]
        )
        structure_right: mw.Rectangle = (
            structures[1] if structure_left is structures[0] else structures[0]
        )

        assert np.isclose(np.min(structure_left.geometry.x_min), -5 - buffer)
        assert np.isclose(np.min(structure_left.geometry.y_min), h_min)
        assert np.isclose(np.max(structure_left.geometry.y_max), h_min + 1)

        assert np.isclose(np.max(structure_right.geometry.x_max), +5 + buffer)
        assert np.isclose(np.min(structure_right.geometry.y_min), h_min)
        assert np.isclose(np.max(structure_right.geometry.y_max), h_min + 1)


def test_gdsf_dumbell_pinched_apart():
    gf.clear_cache()
    gf.gpdk.PDK.activate()

    dumbell = gf.Component()
    left = dumbell << gf.components.rectangle(size=(4, 2), centered=True)
    left.movey(-4)
    right = dumbell << gf.components.rectangle(size=(4, 2), centered=True)
    right.movey(4)
    _bar = dumbell << gf.components.rectangle(size=(0.5, 6), centered=True)

    extrusion_rules = {
        (1, 0): [
            mw.GdsExtrusionRule(material=mw.silicon, h_min=0, h_max=1, buffer=-0.5)
        ]
    }

    structures = mw.extrude_gdsf_component(dumbell, extrusion_rules)
    assert len(structures) == 2

    cell = mw.Cell(
        structures=structures,
        z_min=-1,
        z_max=1,
        mesh=mw.Mesh2D(x=np.linspace(-6, 6, 51), y=np.linspace(-6, 6, 51)),
    )
    structure_left = (
        cell.structures_2d[0]
        if cell.structures_2d[0].geometry.x_min < cell.structures_2d[1].geometry.x_min
        else cell.structures_2d[1]
    )
    structure_right = (
        cell.structures_2d[1]
        if structure_left is cell.structures_2d[0]
        else cell.structures_2d[0]
    )

    assert np.isclose(structure_left.geometry.x_min, -5 + 0.5)

    # there should be a little nub left over from buffering away dumbbell handlebar
    assert -3.5 < structure_left.geometry.x_max < 3
    assert 3 < structure_right.geometry.x_min < 3.5

    assert np.isclose(structure_right.geometry.x_max, 5 - 0.5)


def test_gdspy_extrusion_rectangle():
    import gdspy

    cell = gdspy.Cell("rectangle")
    cell.add(gdspy.Rectangle((0.0, 0.0), (1.0, 2.0), layer=1, datatype=0))
    rule = mw.GdsExtrusionRule(material=mw.silicon, h_min=0, h_max=1)

    structures = mw.extrude_gdspy_cell(cell, {(1, 0): [rule]})

    assert len(structures) == 1
    assert structures[0].material == mw.silicon
    assert isinstance(structures[0].geometry, mw.Prism)

    poly = structures[0].geometry.poly
    assert np.isclose(np.min(poly[:, 0]), 0.0)
    assert np.isclose(np.max(poly[:, 0]), 1.0)
    assert np.isclose(np.min(poly[:, 1]), 0.0)
    assert np.isclose(np.max(poly[:, 1]), 2.0)


def test_gdstk_extrusion_rectangle():
    import gdstk

    cell = gdstk.Cell("rectangle")
    cell.add(gdstk.rectangle((0.0, 0.0), (1.0, 2.0), layer=1, datatype=0))
    rule = mw.GdsExtrusionRule(material=mw.silicon, h_min=0, h_max=1)

    structures = mw.extrude_gdstk_cell(cell, {(1, 0): [rule]})

    assert len(structures) == 1
    assert structures[0].material == mw.silicon
    assert isinstance(structures[0].geometry, mw.Prism)

    poly = structures[0].geometry.poly
    assert np.isclose(np.min(poly[:, 0]), 0.0)
    assert np.isclose(np.max(poly[:, 0]), 1.0)
    assert np.isclose(np.min(poly[:, 1]), 0.0)
    assert np.isclose(np.max(poly[:, 1]), 2.0)


def test_gdspy_hole():
    """Essentially a duplicate of test_gdsf_hole"""
    import gdspy

    cell = gdspy.Cell("hole")
    cell.add(
        gdspy.Polygon(
            [
                (5, -5),
                (5, 5),
                (-5, 5),
                (-5, -5),
                (-3, -3),
                (-3, 3),
                (3, 3),
                (3, -3),
                (-3, -3),
                (-5, -5),
            ],
            layer=1,
            datatype=0,
        )
    )

    extrusion_rules = {
        (1, 0): [
            mw.GdsExtrusionRule(
                material=mw.silicon_oxide,
                h_min=0,
                h_max=1,
                buffer=0.5,
            ),
            mw.GdsExtrusionRule(
                material=mw.silicon_nitride,
                h_min=1,
                h_max=2,
                buffer=0,
            ),
            mw.GdsExtrusionRule(
                material=mw.silicon,
                h_min=2,
                h_max=3,
                buffer=-0.5,
            ),
        ]
    }
    structures = mw.extrude_gds(cell, extrusions=extrusion_rules)
    cell_2d = mw.Cell(
        structures=structures,
        z_min=-5,
        z_max=5,
        mesh=mw.Mesh2D(x=np.linspace(-6, 6, 51), y=np.linspace(-6, 6, 51)),
    )

    assert (
        len(cell_2d.structures_2d) == 6
    )  # 3 extrusion rules, each makes 2 structures on each side

    silicon_oxide_structures = [
        s for s in cell_2d.structures_2d if s.material == mw.silicon_oxide
    ]
    silicon_nitride_structures = [
        s for s in cell_2d.structures_2d if s.material == mw.silicon_nitride
    ]
    silicon_structures = [s for s in cell_2d.structures_2d if s.material == mw.silicon]

    for structures, buffer, h_min in [
        (silicon_oxide_structures, 0.5, 0),
        (silicon_nitride_structures, 0, 1),
        (silicon_structures, -0.5, 2),
    ]:
        assert (
            len(structures) == 2
        )  # each extrusion rule makes 2 structures on each side
        structure_left = (
            structures[0]
            if structures[0].geometry.x_min < structures[1].geometry.x_min
            else structures[1]
        )
        structure_right: mw.Rectangle = (
            structures[1] if structure_left is structures[0] else structures[0]
        )

        assert np.isclose(np.min(structure_left.geometry.x_min), -5 - buffer)
        assert np.isclose(np.min(structure_left.geometry.y_min), h_min)
        assert np.isclose(np.max(structure_left.geometry.y_max), h_min + 1)

        assert np.isclose(np.max(structure_right.geometry.x_max), +5 + buffer)
        assert np.isclose(np.min(structure_right.geometry.y_min), h_min)
        assert np.isclose(np.max(structure_right.geometry.y_max), h_min + 1)


def test_gdstk_hole():
    """Essentially a duplicate of test_gdsf_hole"""
    import gdstk

    cell = gdstk.Cell("hole")
    cell.add(
        gdstk.Polygon(
            [
                (5, -5),
                (5, 5),
                (-5, 5),
                (-5, -5),
                (-3, -3),
                (-3, 3),
                (3, 3),
                (3, -3),
                (-3, -3),
                (-5, -5),
            ],
            layer=1,
            datatype=0,
        )
    )

    extrusion_rules = {
        (1, 0): [
            mw.GdsExtrusionRule(
                material=mw.silicon_oxide,
                h_min=0,
                h_max=1,
                buffer=0.5,
            ),
            mw.GdsExtrusionRule(
                material=mw.silicon_nitride,
                h_min=1,
                h_max=2,
                buffer=0,
            ),
            mw.GdsExtrusionRule(
                material=mw.silicon,
                h_min=2,
                h_max=3,
                buffer=-0.5,
            ),
        ]
    }
    structures = mw.extrude_gds(cell, extrusions=extrusion_rules)
    cell_2d = mw.Cell(
        structures=structures,
        z_min=-5,
        z_max=5,
        mesh=mw.Mesh2D(x=np.linspace(-6, 6, 51), y=np.linspace(-6, 6, 51)),
    )

    assert (
        len(cell_2d.structures_2d) == 6
    )  # 3 extrusion rules, each makes 2 structures on each side

    silicon_oxide_structures = [
        s for s in cell_2d.structures_2d if s.material == mw.silicon_oxide
    ]
    silicon_nitride_structures = [
        s for s in cell_2d.structures_2d if s.material == mw.silicon_nitride
    ]
    silicon_structures = [s for s in cell_2d.structures_2d if s.material == mw.silicon]

    for structures, buffer, h_min in [
        (silicon_oxide_structures, 0.5, 0),
        (silicon_nitride_structures, 0, 1),
        (silicon_structures, -0.5, 2),
    ]:
        assert (
            len(structures) == 2
        )  # each extrusion rule makes 2 structures on each side
        structure_left = (
            structures[0]
            if structures[0].geometry.x_min < structures[1].geometry.x_min
            else structures[1]
        )
        structure_right: mw.Rectangle = (
            structures[1] if structure_left is structures[0] else structures[0]
        )

        assert np.isclose(np.min(structure_left.geometry.x_min), -5 - buffer)
        assert np.isclose(np.min(structure_left.geometry.y_min), h_min)
        assert np.isclose(np.max(structure_left.geometry.y_max), h_min + 1)

        assert np.isclose(np.max(structure_right.geometry.x_max), +5 + buffer)
        assert np.isclose(np.min(structure_right.geometry.y_min), h_min)
        assert np.isclose(np.max(structure_right.geometry.y_max), h_min + 1)


@pytest.fixture(scope="module", autouse=True)
def cleanup_global_state():
    # Setup code (runs once before the first test in this file)
    yield
    # Teardown code (runs once after the last test in this file completes)
    gf.clear_cache()
    gf.gpdk.PDK.activate()  # reactivate for other tests
